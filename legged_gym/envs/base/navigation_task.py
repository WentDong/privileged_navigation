# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
from itertools import product
from time import time
from typing import Dict, Tuple
from warnings import WarningMessage

import numpy as np
import torch
from isaacgym import gymapi, gymtorch, gymutil
from isaacgym.torch_utils import *
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.helpers import class_to_dict, get_load_path
from legged_gym.utils.math_utils import (coordinate_transform,
                                         estimate_next_pose, quat_apply_yaw,
                                         torch_rand_sqrt_float, wrap_to_pi)
# from legged_gym.utils.maze_solver import MazeSolver
from legged_gym.utils.task_registry import task_registry
from legged_gym.utils.terrain import Terrain
from rsl_rl.datasets.motion_loader import AMPLoader
from rsl_rl.modules.actor_critic import ActorCritic
from rsl_rl.modules.actor_critic_teacher import TeacherActorCritic
from rsl_rl.modules.actor_critic_student import StudentActorCritic
from torch import Tensor

from ..a1.a1_navigation_config import A1NavigationCfg, A1NavigationCfgPPO, A1LocomotionCfgPPO
from .legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

COM_OFFSET = torch.tensor([0.012731, 0.002186, 0.000515])
HIP_OFFSETS = torch.tensor([
    [0.183, 0.047, 0.],
    [0.183, -0.047, 0.],
    [-0.183, 0.047, 0.],
    [-0.183, -0.047, 0.]]) + COM_OFFSET


class NavigationTask(BaseTask):
    def __init__(self, cfg: A1NavigationCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = True
        self.init_done = False
        self._parse_cfg(self.cfg)
        
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.num_locomotion_actions = self.cfg.locomotion.num_actions
        
        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_locomotion_model()
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True
        self.navi_curriculum_level = 0
        self.success_buf = torch.zeros((self.num_envs, ), device=self.device, dtype=torch.bool)
        if self.cfg.env.reference_state_initialization:
            self.amp_loader = AMPLoader(motion_files=self.cfg.env.amp_motion_files, device=self.device, time_between_frames=self.dt)

        if self.locomotion_cfg.env.include_history_steps is not None:
            self.locomotion_obs_buf_history = gymutil.EpisodeHistoryBuffer(self.num_envs, self.locomotion_cfg.env.num_observations-self.locomotion_cfg.env.privileged_dim - self.locomotion_cfg.env.height_dim, self.locomotion_cfg.env.include_history_steps, self.device)
        self.locomotion_obs_buf = torch.zeros(self.num_envs, self.locomotion_cfg.env.num_observations-self.locomotion_cfg.env.privileged_dim - self.locomotion_cfg.env.height_dim, device=self.device, dtype=torch.float)

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        if self.cfg.env.include_history_steps is not None:
            self.obs_buf_history.reset(
                torch.arange(self.num_envs, device=self.device),
                self.obs_buf[torch.arange(self.num_envs, device=self.device)])

        if hasattr(self.cfg.env, "include_privileged_history_steps") and self.cfg.env.include_privileged_history_steps is not None:
            self.privileged_obs_buf_history.reset(
                torch.arange(self.num_envs, device=self.device),
                self.privileged_obs_buf[torch.arange(self.num_envs, device=self.device)])
        
        if self.locomotion_cfg.env.include_history_steps is not None:
            self.locomotion_obs_buf_history.reset(
                torch.arange(self.num_envs, device=self.device),
                self.locomotion_obs_buf[torch.arange(self.num_envs, device=self.device)])

        obs, privileged_obs = self.initial_step()
        self.trajectory_history[:] = 0.
        return obs, privileged_obs

    def initial_step(self):
        locomotion_actions = torch.zeros(self.num_envs, self.num_locomotion_actions, device=self.device, requires_grad=False)
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(locomotion_actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        reset_env_ids, terminal_amp_states = self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.cfg.env.include_history_steps is not None:
            self.obs_buf_history.reset(reset_env_ids, self.obs_buf[reset_env_ids])
            self.obs_buf_history.insert(self.obs_buf)
            policy_obs = self.obs_buf_history.get_obs_vec(np.arange(self.include_history_steps))
        else:
            policy_obs = self.obs_buf
        # print("obs_buf.shape", self.obs_buf.shape, "obs.shape", policy_obs.shape)

        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        # print(hasattr(self.cfg.env, "include_privileged_history_steps"))            
        if hasattr(self.cfg.env, "include_privileged_history_steps") and self.cfg.env.include_privileged_history_steps is not None:
            self.privileged_obs_buf_history.reset(reset_env_ids, self.privileged_obs_buf[reset_env_ids])
            self.privileged_obs_buf_history.insert(self.privileged_obs_buf)
            privileged_obs = self.privileged_obs_buf_history.get_obs_vec(np.arange(self.include_history_steps))
        else:
            privileged_obs = self.privileged_obs_buf

        self.locomotion_obs_buf = torch.clip(self.locomotion_obs_buf, -clip_obs, clip_obs)
        if self.locomotion_cfg.env.include_history_steps is not None:
            reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            self.locomotion_obs_buf_history.reset(reset_env_ids, self.locomotion_obs_buf[reset_env_ids])
            self.locomotion_obs_buf_history.insert(self.locomotion_obs_buf)

        return policy_obs, privileged_obs

    def velocity_clip(self):
        self.commands[:, 0] = torch.clip(self.commands[:, 0], self.command_ranges["lin_vel_x"][0]/self.cfg.commands.commands_scale, self.command_ranges["lin_vel_x"][1]/self.cfg.commands.commands_scale)
        # self.commands[:, 1] = torch.clip(self.commands[:, 1], self.command_ranges["lin_vel_y"][0]/self.cfg.commands.commands_scale, self.command_ranges["lin_vel_y"][1]/self.cfg.commands.commands_scale)
        self.commands[:, 1] *= self.command_ranges["lin_vel_y"][1]
        
        self.commands[:, :2] *= self.cfg.commands.commands_scale

        if self.cfg.commands.heading_command:
            self.commands[:, 3] = torch.clip(self.commands[:, 3], self.command_ranges["heading"][0]/self.cfg.commands.commands_scale, self.command_ranges["heading"][1]/self.cfg.commands.commands_scale)
            self.commands[:, 3] *= self.cfg.commands.commands_scale
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.) 
        else:
            self.commands[:, 2] = torch.clip(self.commands[:, 2], self.command_ranges["ang_vel_yaw"][0]/self.cfg.commands.commands_scale, self.command_ranges["ang_vel_yaw"][1]/self.cfg.commands.commands_scale)
            self.commands[:, 2] *= self.cfg.commands.commands_scale
    def step(self, commands):
        """ Calculate actions by command, apply actions, simulate, 
        call self.pre_physics_step()
        call self.post_physics_step()

        Args:
            commands (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
            Here, commands are actions in navigation RL model.
        """
        
        '''In step, first record the current states which is used to calculate rewards.'''
        # Navigation Task: actions is speed command
        # import pdb; pdb.set_trace()
        # self.root_states = torch.tensor([[ 4.6973e+00,  3.1353e+00,  3.5121e-01, -4.4342e-03,  4.5626e-03,
        #   3.6715e-04,  9.9998e-01, -2.7300e-01, -3.5909e-01,  1.5151e-02,
        #  -8.6031e-01,  6.8883e-01, -1.0073e-01]], device = self.device)
        # self.base_ang_vel = torch.tensor([[-0.8589,  0.6904, -0.1025]], device=self.device)
        # self.projected_gravity = torch.tensor([[ 0.0091,  0.0089, -0.9999]], device = self.device)
        # self.base_lin_vel = torch.tensor([[-0.2734, -0.3590,  0.0095]], device = self.device)
        # self.dof_pos = torch.tensor([[ 0.0038,  0.5186, -1.5367,  0.0171,  0.7761, -1.2443, -0.0102,  0.8915, -1.4719, -0.0097,  1.3407, -1.1256]], device = self.device)
        # self.dof_vel = torch.tensor([[  0.8463,   6.6981,  -1.2024,   0.7976,   2.9476, -11.1295,  -0.1491, 2.6959,  -2.2677,  -0.9284,  -3.7972, -12.3351]], device = self.device)
        self.last_commands[:] = self.commands[:]
        self.commands = commands
        # self.commands[:, 0] = 0.5
        # self.commands[:, 1:] = 0
        self.velocity_clip()
        self.rew_buf[:] = 0. # reset reward buffer.
        self.nav_reset_buf = 0 # reset reset buffer

        '''
        Now we are debugging. We will set the line 215 to 277 as commentory, and check whether the actor can imitate the velocity of (0.5,0,0)
        '''
        # self.compute_locomotion_observations() # Compute locomotion_observations since the update of commands.


        # # print("NEXT POS:", self.cur_pos[0])
        # # print("###########")
        self.last_pos = self.cur_pos.clone().detach().to(self.device)
        # # print("LAST POS:", self.last_pos[0])
        # # print("CCCCC ROOT STATES:", self.root_states[0,:3])
        # # print("::::: IN STEP")
        # # print("commands: ", commands)
        if np.random.rand() < 0.1:
            print("COMMAND: ", self.commands[:5])
        # Calculate speed actions from commands
        '''For a single Command, call 5 locomotion steps.'''
        for it in range(5):
            self.pre_physics_step() # Get locomotion action by pretrained model.
            clip_locomotion_actions = self.cfg.normalization.clip_actions
            self.locomotion_actions = torch.clip(self.locomotion_actions, -clip_locomotion_actions, clip_locomotion_actions).to(self.device)

            # step physics and render each frame
            self.render()
            for _ in range(self.cfg.control.decimation):
                self.torques = self._compute_torques(self.locomotion_actions).view(self.torques.shape)
                self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
                self.gym.simulate(self.sim)
                if self.device == 'cpu':
                    self.gym.fetch_results(self.sim, True)
                self.gym.refresh_dof_state_tensor(self.sim)
                # self.base_quat[:] = self.root_states[:, 3:7]
                # self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
                # self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
                # self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

            # import pdb; pdb.set_trace()
            reset_env_ids, terminal_amp_states = self.post_physics_step()

            # return clipped obs, clipped states (None), rewards, dones and infos
            clip_obs = self.cfg.normalization.clip_observations
            self.locomotion_obs_buf = torch.clip(self.locomotion_obs_buf, -clip_obs, clip_obs)
            if self.locomotion_cfg.env.include_history_steps is not None:
                reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
                self.locomotion_obs_buf_history.reset(reset_env_ids, self.locomotion_obs_buf[reset_env_ids])
                self.locomotion_obs_buf_history.insert(self.locomotion_obs_buf)

            

            # print("EEEEE LAST POS:", self.last_pos[0])
            # print("FFFFF CUR POS:", self.cur_pos[0])
            # print("DDDDD ROOT STATES:", self.root_states[0,:3])
        
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)
        self.compute_reward()
        self.check_success_reset()
        self.nav_reset_buf |= self.success_buf
        clip_obs = self.cfg.normalization.clip_observations

        reset_env_ids = self.nav_reset_buf.nonzero(as_tuple=False).flatten()
        #update obs_buffer
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.cfg.env.include_history_steps is not None:
            self.obs_buf_history.reset(reset_env_ids, self.obs_buf[reset_env_ids])
            self.obs_buf_history.insert(self.obs_buf)
            policy_obs = self.obs_buf_history.get_obs_vec(np.arange(self.include_history_steps))
        else:
            policy_obs = self.obs_buf
        # print("obs_buf.shape", self.obs_buf.shape, "obs.shape", policy_obs.shape)

        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        if hasattr(self.cfg.env, "include_privileged_history_steps") and self.cfg.env.include_privileged_history_steps is not None:
            self.privileged_obs_buf_history.reset(reset_env_ids, self.privileged_obs_buf[reset_env_ids])
            self.privileged_obs_buf_history.insert(self.privileged_obs_buf)
            privileged_obs = self.privileged_obs_buf_history.get_obs_vec(np.arange(self.include_history_steps))
        else:
            privileged_obs = self.privileged_obs_buf
        
        self.locomotion_obs_buf = torch.clip(self.locomotion_obs_buf, -clip_obs, clip_obs)
        if self.locomotion_cfg.env.include_history_steps is not None:
            self.locomotion_obs_buf_history.reset(reset_env_ids, self.locomotion_obs_buf[reset_env_ids])
            self.locomotion_obs_buf_history.insert(self.locomotion_obs_buf)

        if np.random.rand()<0.05:
            print("REWARD:", self.rew_buf[:5])
        # self.nav_reset_buf = torch.ones_like(self.nav_reset_buf, dtype=torch.bool, device=self.device)
        return policy_obs, privileged_obs, self.rew_buf, self.nav_reset_buf, self.extras, reset_env_ids, None

    def get_observations(self):
        if self.cfg.env.include_history_steps is not None:
            policy_obs = self.obs_buf_history.get_obs_vec(np.arange(self.include_history_steps))
        else:
            policy_obs = self.obs_buf
        return policy_obs
    
    def get_privileged_observations(self):
        if hasattr(self.cfg.env, "include_privileged_history_steps") and self.cfg.env.include_privileged_history_steps is not None:
            privileged_obs = self.privileged_obs_buf_history.get_obs_vec(np.arange(self.include_history_steps))
        else:
            privileged_obs = self.privileged_obs_buf
        return privileged_obs
    
    def pre_physics_step(self):
        """ check commands and state, compute actions
        
        """
        # self.base_quat[:] = self.root_states[:, 3:7]
        # self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        # self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        # self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        # self.compute_teacher_commands()
        # self.compute_locomotion_observations()
        
        locomotion_obs = self.get_locomotion_observations()
        obs_without_command = torch.concat((locomotion_obs[:, 0:6],
                                            locomotion_obs[:, 9:]), dim=1)
        # print(locomotion_obs.shape)
        # print("OBS_WITHOUT_COMMAND:", obs_without_command)
        self.trajectory_history = torch.concat((self.trajectory_history[:, 1:], obs_without_command.unsqueeze(1)), dim=1)
        

        # print("locomotion_obs_buf.shape:", self.locomotion_obs_buf.shape)
        # print(self.locomotion_obs_buf)
        history = self.trajectory_history.flatten(1).to(self.device)
        # print(history.detach())
        # print("OBS:", locomotion_obs.detach(), history.detach())
        # print("COMMANDS IN ACT:", locomotion_obs[:,6:9])
        self.locomotion_actions[:] = self.locomotion_policy(locomotion_obs[:, 6:9].detach(), history.detach())[:]
        # print("ACT:", self.locomotion_actions.detach())
        # return actions
    
    def get_locomotion_observations(self):
        if self.locomotion_cfg.env.include_history_steps is not None:
            locomotion_obs = self.locomotion_obs_buf_history.get_obs_vec(np.arange(self.locomotion_cfg.env.include_history_steps))
        else:
            locomotion_obs = self.locomotion_obs_buf
        return locomotion_obs
    
    def compute_locomotion_observations(self):
        commands = self.commands[:]
        # print(commands.shape)
        # print("ANG_VEL:", self.base_ang_vel)
        # print("GRAVITY:", self.projected_gravity)
        # print("DOF_POS:", self.dof_pos)
        # print("DOF_VEL:", self.dof_vel)
        self.locomotion_obs_buf = torch.cat(( 
                                    # self.base_lin_vel * self.obs_scales.lin_vel, # Treate as part privileged_dim in locomotion model, so not included here.
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.locomotion_actions
                                    ),dim=-1)
        # print("LOCOMOTION_OBS_SHAPE:", self.locomotion_obs_buf.shape)
        
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        # self.update_navigation_landmarks()
        self.check_termination()
        self.nav_reset_buf |= self.reset_buf
        self.compute_termination_reward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        terminal_amp_states = self.get_amp_observations()[env_ids]
        self.reset_idx(env_ids)

        # after reset idx, the base_lin_vel, base_ang_vel, projected_gravity, height has changed, so should be re-computed
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.compute_locomotion_observations()

        self.cur_pos = self.root_states[:, :3].clone().detach().to(self.device)
        # self.last_actions[:] = self.actions[:]
        # print("LAST_LOCOMOTION_ACTION: ", self.last_locomotion_actions[0])
        self.last_locomotion_actions[:] = self.locomotion_actions[:]
        # print("CUR_LOCOMOTION_ACTION: ", self.locomotion_actions[0])
        self.last_dof_pos[:] = self.dof_pos[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        self.trajectory_history[env_ids] = 0.
        # import pdb; pdb.set_trace()
        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        return env_ids, terminal_amp_states

    def check_termination(self):
        """ Check if environments need to be reset
        """
        # TBD: Reset due to the goal has been reached
        
        
        # Reset due to time out
        self.out_off_world = self.root_states[:, 2] < -20
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        # self.reset_buf |= self.success_buf
        # self.reset_buf |= self.out_off_world
        # if self.success_buf[0]:
            # print("!!!!!!!!SUCCESS for env0!")
        if self.time_out_buf[0]:
            print("!!!!!!!!TIME OUT for env0!")
        elif self.out_off_world[0]:
            print("!!!!!!!!OUT OFF WORLD for env0!")
        elif self.reset_buf[0]:
            print("!!!!!!!!COLLISON TERMINATION for env0!")
        
        # if self.reset_buf[0]:
        #     print(self.measured_heights[0, :])

    def check_success_reset(self):
        self.success_buf[:] = torch.norm(self.cur_pos - self.task_goals, p=2, dim = 1) < self.cfg.task.success_epsilon
        env_ids = self.success_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) <=0:
            return []
        self.reset_idx(env_ids)
        # after reset idx, the base_lin_vel, base_ang_vel, projected_gravity, height has changed, so should be re-computed
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.compute_locomotion_observations()
        self.trajectory_history[env_ids] = 0.


        return env_ids


    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        # if self.cfg.terrain.curriculum:
        #     self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        # if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
        #     self.update_command_curriculum(env_ids)
        if self.cfg.task.curriculum:
            self.update_navigation_curriculum(env_ids)
        # reset robot states
        
        # Here we add the reset of starting points and goals. Which decides the reset point of the root of the robot.
        self._resample_goals(env_ids)
        self._resample_startings(env_ids)

        # if self.cfg.env.reference_state_initialization:
        #     frames = self.amp_loader.get_full_frame_batch(len(env_ids))
        #     self._reset_dofs_amp(env_ids, frames)
        #     self._reset_root_states_amp(env_ids, frames)
        # else:
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        # self._resample_commands(env_ids)

        if self.cfg.domain_rand.randomize_gains:
            new_randomized_gains = self.compute_randomized_gains(len(env_ids))
            self.randomized_p_gains[env_ids] = new_randomized_gains[0]
            self.randomized_d_gains[env_ids] = new_randomized_gains[1]

        # reset buffers
        # self.last_actions[env_ids] = 0.
        self.last_commands[env_ids] = 0.
        self.last_locomotion_actions[env_ids] = 0.
        self.last_dof_pos[env_ids] = 0
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.success_buf[env_ids] = 0
        self.commands[env_ids,:] = 0. # RESET ENV AND SET THE VELOCITY COMMANDS TO ZERO.

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        # if self.cfg.terrain.curriculum:
        #     self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        
        # if self.cfg.commands.curriculum:
        #     self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        #     self.extras["episode"]["max_command_yaw"] = self.command_ranges["ang_vel_yaw"][1]
        if self.cfg.task.curriculum:
            self.extras["episode"]["navi_level"] = self.navi_curriculum_level

        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    def compute_termination_reward(self):
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
        # if "success" in self.reward_scales:
        #     rew = self._reward_success() * self.reward_scales["success"]
        #     self.rew_buf += rew
        #     self.episode_sums["success"] += rew
        

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            # import pdb; pdb.set_trace()
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        

    def compute_relative_goals(self, with_orientation=False):
        ''' Compute Goal's relative position and orientation
            By robot's quat, the goal should be represented in the robot's coordinates.
            Here the output relative goal is in the local coorinates.
            Maybe just minus the global goal by the robot's position works well also?
            with_orientation: bool, whether to compute the goal in the local coordinates with orientation.
                If it is True, it will return the relative goal with orientation.
                If it is False, it will return the relative goal without orientation (just minus )
        '''
        if with_orientation:
            from legged_gym.utils.math_utils import coordinate_transform_3D
            target_relative_goals = coordinate_transform_3D(self.base_quat, self.root_states[:, :3], self.task_goals)
            return target_relative_goals.to(self.device)
        else:
            return self.task_goals.to(self.device) - self.root_states[:, :3]
        
    def update_depth_buffer(self):
        if not self.cfg.depth.use_camera:
            return

        if self.global_counter % self.cfg.depth.update_interval != 0:
            return
        self.gym.step_graphics(self.sim) # required to render in headless mode
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        for i in range(self.num_envs):
            depth_image_ = self.gym.get_camera_image_gpu_tensor(self.sim, 
                                                                self.envs[i], 
                                                                self.cam_handles[i],
                                                                gymapi.IMAGE_DEPTH)
            
            depth_image = gymtorch.wrap_tensor(depth_image_)
            depth_image = self.process_depth_image(depth_image, i)

            init_flag = self.episode_length_buf <= 1
            if init_flag[i]:
                self.depth_buffer[i] = torch.stack([depth_image] * self.cfg.depth.buffer_len, dim=0)
            else:
                self.depth_buffer[i] = torch.cat([self.depth_buffer[i, 1:], depth_image.to(self.device).unsqueeze(0)], dim=0)

        self.gym.end_access_image_tensors(self.sim)

    def compute_observations(self):
        """ Computes observations
        """
        # Navigation Task: Change if need
        self.privileged_obs_buf = torch.cat((  
                                    # self.base_lin_vel * self.obs_scales.lin_vel,    # 3
                                    # self.base_ang_vel  * self.obs_scales.ang_vel,   # 3 
                                    self.compute_relative_goals(with_orientation=self.cfg.env.observation_with_orientatoin),  # 3
                                    # self.root_states[:, :3],
                                    # self.task_goals[:, :3],
                                    self.projected_gravity, # 3
                                    # self.root_states[:, 3:7], # 4 quat of base
                                    # (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,    # 12
                                    # self.dof_vel * self.obs_scales.dof_vel, #12
                                    # self.locomotion_actions         #12
                                    self.last_pos[:, :3], #3
                                    # self.commands[:, :3] * self.commands_scale,  #3
                                    ),dim=-1)  
        '''Only contains: Goals, orientation, commands, last commands'''
        # print(self.privileged_obs_buf.shape)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)
            # print(heights[0,:10], self.root_states[:,3])
            # print(heights.shape)
            # print(self.privileged_obs_buf.shape)

        # add noise if needed
        if self.add_noise:
            self.privileged_obs_buf += (2 * torch.rand_like(self.privileged_obs_buf) - 1) * self.noise_scale_vec

        # Remove velocity observations from policy observation.
        if self.num_obs == self.num_privileged_obs - 6:
            self.obs_buf = self.privileged_obs_buf[:, 6:]
        elif self.num_obs == self.num_privileged_obs - 3:
            self.obs_buf = self.privileged_obs_buf[:, 3:]
        else:
            self.obs_buf = torch.clone(self.privileged_obs_buf)
    
    
    # def get_navigation_path(self):
    #     """ Call Astar package to get the navigation path
    #         self.navigation_path[i] = [(x0,y0),(x1,y1),...(xn,yn)]
    #     """
    #     self.navigation_path = {}
    #     for i in range(self.num_envs):
    #         heightfield = self.terrain.height_field_raw
            
    #         x0 = int(self.task_startings[i][0].item())
    #         y0 = int(self.task_startings[i][1].item())
    #         xn = int(self.task_goals[i][0].item())
    #         yn = int(self.task_goals[i][1].item())
            
    #         starting = (x0,y0)
    #         goal = (xn,yn)
            
    #         print("Env id", i, "starting", starting, "goal", goal)
            
    #         foundPath = MazeSolver(heightfield).astar(starting, goal)
    #         if foundPath:
    #             self.navigation_path[i] = torch.Tensor([[p[0],p[1]] for p in foundPath])
    #             self.task_next_landmarks[i,:] = self.navigation_path[i][1]
    #         else:
    #             self.navigation_path[i] = None
        
    #     # print(self.navigation_path)
    
    # def update_navigation_landmarks(self):
    #     """ Update navigation landmarks if the robot has reached next landmark.
    #     """
    #     # print("Update navigation landmarks...")
    #     # Here the keys (env_id) are int, not scalar tensor.
    #     for i in range(self.num_envs):
    #         base_pos = self.root_states[i, :3].cpu().numpy()
    #         next_landmark = self.task_next_landmarks[i,:].cpu().numpy()
    #         # print(base_pos,next_landmark)
    #         base_coordinate = base_pos / self.cfg.terrain.horizontal_scale
            
    #         if int(base_coordinate[0]) == int(next_landmark[0]) and int(base_coordinate[1]) == int(next_landmark[1]):
    #             print("Env id", i, "has reached next landmark")
    #             if self.navigation_path[i] is not None:
    #                 if len(self.navigation_path[i]) > 2:
    #                     self.task_next_landmarks[i,:] = self.navigation_path[i][2]
    #                     self.navigation_path[i] = self.navigation_path[i][1:]
    #                 else:
    #                     self.task_next_landmarks[i,:] = self.navigation_path[i][1]
    #                     self.navigation_path[i] = None
    #             else:
    #                 self.task_next_landmarks[i,:] = self.task_goals[i,:]

    def get_amp_observations(self):
        joint_pos = self.dof_pos
        foot_pos = self.foot_positions_in_base_frame(self.dof_pos).to(self.device)
        base_lin_vel = self.base_lin_vel
        base_ang_vel = self.base_ang_vel
        joint_vel = self.dof_vel
        z_pos = self.root_states[:, 2:3] #- torch.mean(self.measured_heights, dim=-1, keepdim=True)
        return torch.cat((joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel, z_pos), dim=-1)

    def get_full_amp_observations(self):
        joint_pos = self.dof_pos
        foot_pos = self.foot_positions_in_base_frame(self.dof_pos).to(self.device)
        base_lin_vel = self.base_lin_vel
        base_ang_vel = self.base_ang_vel
        joint_vel = self.dof_vel
        pos = self.root_states[:, :3] #- torch.mean(self.measured_heights, dim=-1, keepdim=True)
        rot = self.root_states[:, 3:7]
        foot_vel = torch.zeros_like(foot_pos)
        return torch.cat((pos, rot, joint_pos, foot_pos, base_lin_vel, base_ang_vel, joint_vel, foot_vel), dim=-1)

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _init_locomotion_model(self):
        """ Load locomotion model from cfg. We only need actor part of ActorCritic.
            Refer to play.py
        """
        # set path to load model
        locomotion_cfg_class = eval(self.cfg.locomotion.train_cfg_class_name)
        self.locomotion_cfg = locomotion_cfg_class()
        self.locomotion_cfg.runner.resume = True
        self.locomotion_cfg.runner.load_run = self.cfg.locomotion.load_run
        self.locomotion_cfg.runner.checkpoint = self.cfg.locomotion.checkpoint
        locomotion_cfg_dict = class_to_dict(self.locomotion_cfg)
        # load previously trained model
        resume_path = get_load_path(
                                    root=os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', self.cfg.locomotion.experiment_name), 
                                    load_run=self.locomotion_cfg.runner.load_run, 
                                    checkpoint=self.locomotion_cfg.runner.checkpoint
                                    )

        # Imitate rsl_rl.runners.OnPolicyRunners.__init__
        actor_critic_class = eval(self.locomotion_cfg.runner.policy_class_name) # ActorCritic

        if self.locomotion_cfg.env.include_history_steps is not None:
            num_actor_obs = self.locomotion_cfg.env.num_observations * self.locomotion_cfg.env.include_history_steps
        else:
            num_actor_obs = self.locomotion_cfg.env.num_observations

        if self.cfg.locomotion.num_privileged_obs is not None:
            num_critic_obs = self.cfg.locomotion.num_privileged_obs 
        else:
            num_critic_obs = self.cfg.locomotion.num_observations
        self.locomotion_actor_critic: ActorCritic = actor_critic_class( num_actor_obs=num_actor_obs,
                                                        num_critic_obs=num_critic_obs,
                                                        num_actions=self.locomotion_cfg.env.num_actions,
                                                        height_dim = self.locomotion_cfg.env.height_dim,
                                                        privileged_dim = self.locomotion_cfg.env.privileged_dim,
                                                        history_dim = self.locomotion_cfg.env.history_length * (self.locomotion_cfg.env.num_observations -
                                                        self.locomotion_cfg.env.privileged_dim - self.locomotion_cfg.env.height_dim - 3),
                                                        **locomotion_cfg_dict["policy"]).to(self.device)
        print(f"Loading locomotion model from: {resume_path}")
        loaded_dict = torch.load(resume_path, map_location=self.device)
        
        self.locomotion_actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        # self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict']) # need self.alg: PPO
        # self.current_learning_iteration = loaded_dict['iter']
        
        # Imitate rsl_rl.runners.OnPolicyRunners.get_inference_policy
        self.locomotion_actor_critic.eval()
        self.locomotion_actor_critic.to(self.device)
        self.locomotion_policy = self.locomotion_actor_critic.act_inferenc_without_privileged
    
        """
        Tips: In ActorCritic:
        def act_inference(self, observations):
            actions_mean = self.actor(observations)
            return actions_mean
        """
    
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            # if env_id==0:
            #     # prepare friction randomization
            #     friction_range = self.cfg.domain_rand.friction_range
            #     num_buckets = 64
            #     bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
            #     friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
            #     self.friction_coeffs = friction_buckets[bucket_ids]
            #
            # for s in range(len(props)):
            #     props[s].friction = self.friction_coeffs[env_id]
            rng = self.cfg.domain_rand.friction_range
            self.randomized_frictions[env_id] = np.random.uniform(rng[0], rng[1])
            for s in range(len(props)):
                props[s].friction = self.randomized_frictions[env_id]

        # if self.cfg.domain_rand.randomize_restitution:
        if hasattr(self.cfg.domain_rand, "randomize_restitution") and self.cfg.domain_rand.randomize_restitution:
            rng = self.cfg.domain_rand.restitution_range
            self.randomized_restitutions[env_id] = np.random.uniform(rng[0], rng[1])
            for s in range(len(props)):
                props[s].restitution = self.randomized_restitutions[env_id]
        return props


    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            added_mass = np.random.uniform(rng[0], rng[1])
            self.randomized_added_masses[env_id] = added_mass
            props[0].mass += added_mass

        # randomize com position
        # if self.cfg.domain_rand.randomize_com_pos:
        #     rng = self.cfg.domain_rand.com_pos_range
        #     com_pos = np.random.uniform(rng[0], rng[1])
        #     self.randomized_com_pos[env_id] = com_pos
        #     props[0].com =  gymapi.Vec3(com_pos,0,0)

        # if self.cfg.domain_rand.randomize_link_mass:
        #     rng = self.cfg.domain_rand.link_mass_range
        #     for i in range(1, len(props)):
        #         props[i].mass = props[i].mass * np.random.uniform(rng[0], rng[1])

        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # ACTUALLY IT SHOULD BE NO USE FOR NAVIGATION TASK except for push_robots.

        # Navigation Task：Do not resample commands here.
        # env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        # self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.) 

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    # def _resample_commands(self, env_ids):
    #     """ Randommly select commands of some environments

    #     Args:
    #         env_ids (List[int]): Environments ids for which new commands are needed
    #     """
        
    #     for env_id in env_ids:
    #         self.commands[env_id,:2] = coordinate_transform(self.task_startings[env_id,:],self.task_goals[env_id,:2])
        
        
    #     # self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
    #     # self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
    #     # if self.cfg.commands.heading_command:
    #     #     self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
    #     # else:
    #     #     self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

    #     # set small commands to zero
    #     # self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _resample_startings(self, env_ids):
        """ Randommly select startings of some environments, scale: m
        
        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        
        if len(env_ids) == 0:
            return
        
        # Select new startings, scale: m
        starting_incerment_x = torch_rand_float(self.task_ranges["starting_x"][0], self.task_ranges["starting_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        starting_incerment_y = torch_rand_float(self.task_ranges["starting_y"][0], self.task_ranges["starting_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.task_startings_yaw[env_ids] = torch_rand_float(self.task_ranges["starting_yaw"][0], self.task_ranges["starting_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        # self.task_startings[env_ids, 2] = self.env_origins[env_ids, 2] + 0.3
        
        
        self.task_startings[env_ids, 0] = self.env_origins[env_ids, 0] + starting_incerment_x
        self.task_startings[env_ids, 1] = self.env_origins[env_ids, 1] + starting_incerment_y
        
        px = ((self.task_startings[env_ids, 0] + self.terrain.cfg.border_size)/self.terrain.cfg.horizontal_scale).long().view(-1)
        py = ((self.task_startings[env_ids, 1] + self.terrain.cfg.border_size)/self.terrain.cfg.horizontal_scale).long().view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        self.task_startings[env_ids, 2] = heights.view(len(env_ids)) * self.terrain.cfg.vertical_scale + 0.3
        # Check if new startings are valid
        # valid_starting_mask = map(self._is_valid_starting, env_ids)
        # env_ids_to_resample = filter(lambda x: not x[1], zip(env_ids, valid_starting_mask))
        # env_ids_to_resample = [x[0] for x in env_ids_to_resample]
        
        
        # # If not vaild, regenerate a starting
        # self._resample_startings(env_ids_to_resample)
        
        
    # def _is_valid_starting(self, env_id):
    #     """ Check if the starting is valid: starting should not be in the obstacles.
        
    #     Args:
    #         env_id (int): ID of environment to be checked.
    #     """
    #     start_x, end_x, start_y, end_y = self.meter_to_index(self.task_startings[env_id],self.cfg.task.robot_collision_box)
        
    #     if np.max(self.terrain.height_field_raw[start_x: end_x, start_y: end_y]) > 0:
    #         if self.cfg.task.show_checking:
    #             print(f"In checking starting of env {env_id}, starting is in obstacles.")
    #         return False
    #     else:
    #         if self.cfg.task.show_checking:
    #             print(f"Set a valid starting of env {env_id} at {self.task_startings[env_id]}.")
    #         return True
        
    def _resample_goals(self, env_ids):
        """ Randommly select goals of some environments, scale: m
            In practice, we sample a goal and take the distance between the goal and the starting as the command.
        """
        
        if len(env_ids) == 0:
            return
        # print(len(env_ids))
        goal_incerment_x = torch_rand_float(self.task_ranges["goal_x"][0], self.task_ranges["goal_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        goal_incerment_y = torch_rand_float(self.task_ranges["goal_y"][0], self.task_ranges["goal_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        self.task_goals[env_ids, 0] = self.env_origins[env_ids, 0] + goal_incerment_x
        self.task_goals[env_ids, 1] = self.env_origins[env_ids, 1] + goal_incerment_y

        px = ((self.task_goals[env_ids, 0] + self.terrain.cfg.border_size)/self.terrain.cfg.horizontal_scale).long().view(-1)
        py = ((self.task_goals[env_ids, 1] + self.terrain.cfg.border_size)/self.terrain.cfg.horizontal_scale).long().view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        self.task_goals[env_ids, 2] = heights.view(len(env_ids)) * self.terrain.cfg.vertical_scale + 0.3
        # valid_goal_mask = map(self._is_valid_goal, env_ids)
        # env_ids_to_resample = filter(lambda x: not x[1], zip(env_ids, valid_goal_mask))
        # env_ids_to_resample = [x[0] for x in env_ids_to_resample]
        
        # # If not vaild, regenerate a goal
        # self._resample_goals(env_ids_to_resample)
        
    # def _is_valid_goal(self, env_id):
    #     """ Check if the goal is valid: goal should not be in the obstacles, and goals should have a path to startings. If a valid path found, add it to the table.
        
    #     Calls MazeSolver.astar() to check if a path exists.
        
    #     Args:
    #         env_id (int): ID of environment to be checked.
    #     """
    #     # Check obstacle around
    #     start_x, end_x, start_y, end_y = self.meter_to_index(self.task_goals[env_id],self.cfg.task.robot_collision_box)
        
    #     if np.max(self.terrain.height_field_raw[start_x: end_x, start_y: end_y]) > 0:
    #         if self.cfg.task.show_checking:
    #             print(f"In checking goal of env {env_id}, goal is in obstacles.")
    #         return False

    #     # Check if there is a path from starting to goal
    #     x0, y0 = self.meter_to_index(self.task_startings[env_id])
    #     xn, yn = self.meter_to_index(self.task_goals[env_id])
        
    #     pathfound = MazeSolver(self.terrain.height_field_raw).astar((x0,y0), (xn,yn))
        
    #     if pathfound is None:
    #         if self.cfg.task.show_checking:
    #             print(f"In checking goal of env {env_id}, no path found.")
    #         return False
        
    #     pathfound = list(pathfound)
        
    #     # In self.reset(), env_ids is a tensor of size 0. The key of a dict is better to be an int anyway.
    #     if type(env_id) == torch.Tensor:
    #         if env_id.numel() == 1:
    #             env_id = env_id.item()
    #         else:
    #             raise TypeError("env_id should be a scalar, but got", env_id, "with size",env_id.size())
        
    #     if len(pathfound) < self.cfg.task.min_path_length:
    #         if self.cfg.task.show_checking:
    #             print(f"In checking goal of env {env_id}, too close goal.")
    #         return False
    #     else:
    #         self.navigation_path[env_id] = torch.Tensor([[p[0],p[1]] for p in pathfound])     
    #         if self.cfg.task.show_checking:
    #             print(f"Set a valid goal of env {env_id} at {self.task_goals[env_id]}, have a path of length {len(pathfound)}.")
    #         return True
        
    # def meter_to_index(self, meter_coordinate, area=None):
    #     """ Transfer meter representation to index in height_field_raw.
    #     """
    #     x = meter_coordinate[0]
    #     y = meter_coordinate[1]
        
    #     border_idx = int(self.cfg.terrain.border_size / self.cfg.terrain.horizontal_scale)
    #     x_idx = border_idx + int(x / self.cfg.terrain.horizontal_scale)
    #     y_idx = border_idx + int(x / self.cfg.terrain.horizontal_scale)
        
    #     if area is None:
    #         return x_idx, y_idx
    #     elif isinstance(area, int):
    #         area_incre = int(area / self.cfg.terrain.horizontal_scale)
    #         return x_idx - area_incre, x_idx + area_incre, y_idx - area_incre, y_idx + area_incre
    #     elif isinstance(area, tuple) or isinstance(area, list):
    #         area_incre_x = int(area[0] / self.cfg.terrain.horizontal_scale)
    #         area_incre_y = int(area[1] / self.cfg.terrain.horizontal_scale)
    #         return x_idx - area_incre_x, x_idx + area_incre_x, y_idx - area_incre_y, y_idx + area_incre_y
    #     else:
    #         raise NotImplementedError
    
    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type

        if self.cfg.domain_rand.randomize_gains:
            p_gains = self.randomized_p_gains
            d_gains = self.randomized_d_gains
        else:
            p_gains = self.p_gains
            d_gains = self.d_gains

        if control_type=="P":
            torques = p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - d_gains*self.dof_vel
        elif control_type=="V":
            torques = p_gains*(actions_scaled - self.dof_vel) - d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    # Navigation Task: 不需要用dof，暂不修改
    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    # Navigation Task: 不需要用dof，暂不修改
    # def _reset_dofs_amp(self, env_ids, frames):
    #     """ Resets DOF position and velocities of selected environmments
    #     Positions are randomly selected within 0.5:1.5 x default positions.
    #     Velocities are set to zero.

    #     Args:
    #         env_ids (List[int]): Environemnt ids
    #         frames: AMP frames to initialize motion with
    #     """
    #     self.dof_pos[env_ids] = AMPLoader.get_joint_pose_batch(frames)
    #     self.dof_vel[env_ids] = AMPLoader.get_joint_vel_batch(frames)
    #     env_ids_int32 = env_ids.to(dtype=torch.int32)
    #     self.gym.set_dof_state_tensor_indexed(self.sim,
    #                                           gymtorch.unwrap_tensor(self.dof_state),
    #                                           gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    # Navigation Task: state应包含height map?
    # height map 独立出来了
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            # self.root_states[env_ids, :3] += self.env_origins[env_ids]
            # self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
            # self.root_states[env_ids, :2] += torch_rand_float(-2., 2., (len(env_ids), 2), device=self.device) 
            self._resample_startings(env_ids)
            self._resample_goals(env_ids)
            self.root_states[env_ids, :3] = self.task_startings[env_ids, :3]
            self.root_states[env_ids, 3:7] = quat_from_euler_xyz(
                torch.zeros(len(env_ids)).to(self.device),
                torch.zeros(len(env_ids)).to(self.device),
                self.task_startings_yaw[env_ids],
                )
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]

        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.cur_pos[env_ids,:] = torch.tensor(self.root_states[env_ids, :3]).to(self.device)
        self.last_pos[env_ids, :] = torch.tensor(self.root_states[env_ids, :3]).to(self.device)


    # def _reset_root_states_amp(self, env_ids, frames):
    #     """ Resets ROOT states position and velocities of selected environmments
    #         Sets base position based on the curriculum
    #         Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
    #     Args:
    #         env_ids (List[int]): Environemnt ids
    #     """
    #     # base position
    #     root_pos = AMPLoader.get_root_pos_batch(frames)
    #     root_pos[:, :2] = root_pos[:, :2] + self.env_origins[env_ids, :2]
    #     self.root_states[env_ids, :3] = root_pos
    #     root_orn = AMPLoader.get_root_rot_batch(frames)
    #     self.root_states[env_ids, 3:7] = root_orn
    #     self.root_states[env_ids, 7:10] = quat_rotate(root_orn, AMPLoader.get_linear_vel_batch(frames))
    #     self.root_states[env_ids, 10:13] = quat_rotate(root_orn, AMPLoader.get_angular_vel_batch(frames))

    #     env_ids_int32 = env_ids.to(dtype=torch.int32)
    #     self.gym.set_actor_root_state_tensor_indexed(self.sim,
    #                                                  gymtorch.unwrap_tensor(self.root_states),
    #                                                  gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    #     self.cur_pos[env_ids,:] = torch.tensor(self.root_states[env_ids, :3]).to(self.device)
    #     self.last_pos[env_ids, :] = torch.tensor(self.root_states[env_ids, :3]).to(self.device)
    # def _blocked_root_position(self, env_ids, pos):
    #     """ Return whether the spawn point of the robot is blocked by terrain.
    #     """
    #     # print(self.terrain.height_field_raw.shape) # 820,820
    #     # print(self.terrain.length_per_env_pixels, self.terrain.width_per_env_pixels) # 80,80
        
    #     i = env_ids // self.cfg.terrain.num_rows
    #     j = env_ids % self.cfg.terrain.num_cols
        
    #     length_box = 10
    #     width_box = 10
        
    #     center_x = int(pos[0]*10)
    #     center_y = int(pos[1]*10)
        
    #     # map coordinate system
    #     start_x = center_x-length_box + i * self.terrain.length_per_env_pixels
    #     end_x = center_x+length_box + i * self.terrain.length_per_env_pixels
    #     start_y = center_y-width_box + j * self.terrain.width_per_env_pixels
    #     end_y = center_y+width_box +  j * self.terrain.width_per_env_pixels
        
    #     # self.terrain.height_field_raw[start_x:end_x, start_y:end_y] = 5.
        
    #     return np.max(self.terrain.height_field_raw[start_x: end_x, start_y:end_y]) > 0.01
    #     # return False
    
    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        return
        # Navigation task does not offers a command curriculum up to now.
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if (torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * \
                self.reward_scales["tracking_lin_vel"]) and \
                (torch.mean(self.episode_sums["tracking_ang_vel"][env_ids]) / self.max_episode_length > 0.7 * \
                self.reward_scales["tracking_ang_vel"]):
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.1,
                                                          -self.cfg.commands.max_lin_vel_x_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.1, 0.,
                                                          self.cfg.commands.max_lin_vel_x_curriculum)
            self.command_ranges["lin_vel_y"][0] = np.clip(self.command_ranges["lin_vel_y"][0] - 0.1,
                                                          -self.cfg.commands.max_lin_vel_y_curriculum, 0.)
            self.command_ranges["lin_vel_y"][1] = np.clip(self.command_ranges["lin_vel_y"][1] + 0.1, 0.,
                                                          self.cfg.commands.max_lin_vel_y_curriculum)

            self.command_ranges["ang_vel_yaw"][0] = np.clip(self.command_ranges["ang_vel_yaw"][0] - 0.1,
                                                          -self.cfg.commands.max_ang_vel_yaw_curriculum, 0.)
            self.command_ranges["ang_vel_yaw"][1] = np.clip(self.command_ranges["ang_vel_yaw"][1] + 0.1, 0.,
                                                          self.cfg.commands.max_ang_vel_yaw_curriculum)

    def update_navigation_curriculum(self, env_ids):
        """ Implements a curriculum of increasing difficulties of navigation tasks. 
            1. Increase the range of starting positions.
            2. Decrease the epsilon of success checking.
            3. Increase the range of goals.
            When the average success count increase at 100 for each environment, increase the curriculum level by 1.
        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # print("Success Rate:", torch.mean(self.episode_sums["success"][env_ids])/self.reward_scales["success"])
        
        # print(self.episode_sums["success"].shape)
        # print(self.success_buf.shape)
        # print(self.success_count.shape)
        # print(env_ids.shape)
        self.success_count += self.success_buf[env_ids].sum()
        self.reset_count += len(env_ids)
        # print(self.success_count.shape)
        if self.reset_count < 10000:
            # print("TOO FEW EPISODES!")
            return
        if self.reset_count>50000:
            self.reset_count = 0
            self.success_count = 0
            return
        ratio = self.success_count/self.reset_count
        if  ratio > 0.8:
            self.reset_count = 0
            self.success_count = 0
            self.navi_curriculum_level += 1
            self.task_ranges["starting_x"][0] = np.clip(self.task_ranges["starting_x"][0] - 0.5,
                                                          -self.cfg.task.curriculum_range.max_starting_xy_curriculum, 0.)
            self.task_ranges["starting_x"][1] = np.clip(self.task_ranges["starting_x"][1] + 0.5,
                                                          0., self.cfg.task.curriculum_range.max_starting_xy_curriculum)
            self.task_ranges["starting_y"][0] = np.clip(self.task_ranges["starting_y"][0] - 0.5,
                                                          -self.cfg.task.curriculum_range.max_starting_xy_curriculum, 0.)
            self.task_ranges["starting_y"][1] = np.clip(self.task_ranges["starting_y"][1] + 0.5,
                                                          0., self.cfg.task.curriculum_range.max_starting_xy_curriculum)
            self.task_ranges["goal_x"][0] = np.clip(self.task_ranges["goal_x"][0] - 0.5,
                                                          -self.cfg.task.curriculum_range.max_goal_xy_curriculum, 0.)
            self.task_ranges["goal_x"][1] = np.clip(self.task_ranges["goal_x"][1] + 0.5,
                                                          0., self.cfg.task.curriculum_range.max_goal_xy_curriculum)
            self.task_ranges["goal_y"][0] = np.clip(self.task_ranges["goal_y"][0] - 0.5,
                                                            -self.cfg.task.curriculum_range.max_goal_xy_curriculum, 0.)
            self.task_ranges["goal_y"][1] = np.clip(self.task_ranges["goal_y"][1] + 0.5,
                                                            0., self.cfg.task.curriculum_range.max_goal_xy_curriculum)
            self.cfg.task.success_epsilon = max(self.cfg.task.success_epsilon - 0.05, self.cfg.task.curriculum_range.min_success_epsilon)
            print("UP LEVEL! :)")
            print(f"Update Success epsilon: {self.cfg.task.success_epsilon}")
            print(f"Update Starting range: {self.task_ranges['starting_x']}, {self.task_ranges['starting_y']}")
            print(f"Update Goal range: {self.task_ranges['goal_x']}, {self.task_ranges['goal_y']}")
        elif ratio < 0.4:
            self.reset_count = 0
            self.success_count = 0
            self.navi_curriculum_level -= 1
            self.task_ranges["starting_x"][0] = np.clip(self.task_ranges["starting_x"][0] + 0.5,
                                                          -self.cfg.task.curriculum_range.max_starting_xy_curriculum, 0.)
            self.task_ranges["starting_x"][1] = np.clip(self.task_ranges["starting_x"][1] - 0.5,
                                                          0., self.cfg.task.curriculum_range.max_starting_xy_curriculum)
            self.task_ranges["starting_y"][0] = np.clip(self.task_ranges["starting_y"][0] + 0.5,
                                                          -self.cfg.task.curriculum_range.max_starting_xy_curriculum, 0.)
            self.task_ranges["starting_y"][1] = np.clip(self.task_ranges["starting_y"][1] - 0.5,
                                                          0., self.cfg.task.curriculum_range.max_starting_xy_curriculum)
            self.task_ranges["goal_x"][0] = np.clip(self.task_ranges["goal_x"][0] + 0.5,
                                                          -self.cfg.task.curriculum_range.max_goal_xy_curriculum, 0.)
            self.task_ranges["goal_x"][1] = np.clip(self.task_ranges["goal_x"][1] - 0.5,
                                                          0., self.cfg.task.curriculum_range.max_goal_xy_curriculum)
            self.task_ranges["goal_y"][0] = np.clip(self.task_ranges["goal_y"][0] + 0.5,
                                                            -self.cfg.task.curriculum_range.max_goal_xy_curriculum, 0.)
            self.task_ranges["goal_y"][1] = np.clip(self.task_ranges["goal_y"][1] - 0.5,
                                                            0., self.cfg.task.curriculum_range.max_goal_xy_curriculum)
            self.cfg.task.success_epsilon += 0.05
            print("DOWN LEVEL! :( ")
            print(f"Update Success epsilon: {self.cfg.task.success_epsilon}")
            print(f"Update Starting range: {self.task_ranges['starting_x']}, {self.task_ranges['starting_y']}")
            print(f"Update Goal range: {self.task_ranges['goal_x']}, {self.task_ranges['goal_y']}")

        print(f"ratio: succ/ALL={ratio}. Current Curriculum Level:{self.navi_curriculum_level}")
    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.privileged_obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] =noise_scales.base_pos * noise_level # Relative Goal
        # noise_vec[3:6] = noise_scales.base_pos * noise_level # Current_pos
        # noise_vec[6:9] = 0   # Target Goal
        noise_vec[3:6] = noise_scales.gravity * noise_level
        # noise_vec[12:16] = noise_scales.quat * noise_level# Quat
        noise_vec[6:9] = noise_scales.base_pos * noise_level # Last Pos
        # noise_vec[16:18] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel # last command lin_vel
        # noise_vec[18] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel # last command ang_vel

        # noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        # noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        # noise_vec[6:9] = noise_scales.gravity * noise_level
        # noise_vec[9:12] = 0. # commands
        # noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        # noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        # noise_vec[36:48] = 0. # previous actions
        # if self.cfg.terrain.measure_heights:
        #     noise_vec[48:235] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        if self.cfg.terrain.measure_heights:
            noise_vec[9:185] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
                
        return noise_vec

        

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        # self.gym.refresh_force_sensor_tensor(self.sim)
        # force_sensor_readings = gymtorch.wrap_tensor(sensor_tensor)
        # self.sensor_forces = force_sensor_readings.view(self.num_envs, 4, 6)[..., :3]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[..., 0:3]
        self.rigid_body_lin_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[...,7:10]

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_locomotion_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_locomotion_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_locomotion_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading!
        self.last_commands = torch.zeros_like(self.commands, dtype=torch.float, device=self.device, requires_grad=False)    
        # self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        # self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.locomotion_actions = torch.zeros(self.num_envs, self.num_locomotion_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_locomotion_actions = torch.zeros(self.num_envs, self.num_locomotion_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.cur_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_pos = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.nav_reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.success_count = 0
        self.reset_count = 0

        # self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        if self.cfg.domain_rand.randomize_gains:
            self.randomized_p_gains, self.randomized_d_gains = self.compute_randomized_gains(self.num_envs)
            
        # self.command_pool = torch.Tensor(list(product(
        #     self.cfg.commands.choices.lin_vel_x,
        #     self.cfg.commands.choices.lin_vel_y,
        #     self.cfg.commands.choices.ang_vel_yaw
        #     )))
        
        self.trajectory_history = torch.zeros(size=(self.num_envs, self.locomotion_cfg.env.history_length, self.locomotion_cfg.env.num_observations -
                                            self.locomotion_cfg.env.privileged_dim - self.locomotion_cfg.env.height_dim - 3), device = self.device)
        

    def compute_randomized_gains(self, num_envs):
        p_mult = ((
            self.cfg.domain_rand.stiffness_multiplier_range[0] -
            self.cfg.domain_rand.stiffness_multiplier_range[1]) *
            torch.rand(num_envs, self.num_locomotion_actions, device=self.device) +
            self.cfg.domain_rand.stiffness_multiplier_range[1]).float()
        d_mult = ((
            self.cfg.domain_rand.damping_multiplier_range[0] -
            self.cfg.domain_rand.damping_multiplier_range[1]) *
            torch.rand(num_envs, self.num_locomotion_actions, device=self.device) +
            self.cfg.domain_rand.damping_multiplier_range[1]).float()
        
        return p_mult * self.p_gains, d_mult * self.d_gains


    def foot_position_in_hip_frame(self, angles, l_hip_sign=1):
        theta_ab, theta_hip, theta_knee = angles[:, 0], angles[:, 1], angles[:, 2]
        l_up = 0.2
        l_low = 0.2
        l_hip = 0.08505 * l_hip_sign
        leg_distance = torch.sqrt(l_up**2 + l_low**2 +
                                2 * l_up * l_low * torch.cos(theta_knee))
        eff_swing = theta_hip + theta_knee / 2

        off_x_hip = -leg_distance * torch.sin(eff_swing)
        off_z_hip = -leg_distance * torch.cos(eff_swing)
        off_y_hip = l_hip

        off_x = off_x_hip
        off_y = torch.cos(theta_ab) * off_y_hip - torch.sin(theta_ab) * off_z_hip
        off_z = torch.sin(theta_ab) * off_y_hip + torch.cos(theta_ab) * off_z_hip
        return torch.stack([off_x, off_y, off_z], dim=-1)

    def foot_positions_in_base_frame(self, foot_angles):
        foot_positions = torch.zeros_like(foot_angles)
        for i in range(4):
            foot_positions[:, i * 3:i * 3 + 3].copy_(
                self.foot_position_in_hip_frame(foot_angles[:, i * 3: i * 3 + 3], l_hip_sign=(-1)**(i)))
        foot_positions = foot_positions + HIP_OFFSETS.reshape(12,).to(self.device)
        return foot_positions

    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        # print(self.reward_functions)
        # import pdb; pdb.set_trace()
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldProperties()
        hf_params.column_scale = self.terrain.horizontal_scale
        hf_params.row_scale = self.terrain.horizontal_scale
        hf_params.vertical_scale = self.terrain.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.border_size 
        hf_params.transform.p.y = -self.terrain.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        
        # initialize variants for giving a privileged navigation
        self.task_startings = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.task_startings_yaw = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.task_goals = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        # self.teacher_commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        # self.task_next_landmarks = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self._get_env_origins()

        self._resample_startings(range(self.num_envs))
        self._resample_goals(range(self.num_envs))

        # self._get_task_goals()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []

        #for domain randomization
        self.randomized_frictions = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)
        self.randomized_restitutions = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)
        self.randomized_added_masses = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)
        self.randomized_com_pos = torch.zeros(self.num_envs, 1, device=self.device, requires_grad=False)

        # Before set the starting and goals, generate the agent as default setting (start at the origin)
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.task_startings[i].clone() * self.cfg.terrain.horizontal_scale

            # pos[:2] += torch_rand_float(-2., 2., (2,1), device=self.device).squeeze(1)


            # while self._blocked_root_position(i,pos):
            #     pos = self.env_origins[i].clone()
            #     pos[:2] += torch_rand_float(-2., 2., (2,1), device=self.device).squeeze(1)
            
            # self.task_startings[i,:] = pos / self.cfg.terrain.horizontal_scale
            start_pose.p = gymapi.Vec3(*pos)
            
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            anymal_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "anymal", i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, anymal_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, anymal_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, anymal_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(anymal_handle)
            
            # print('Origin at',self.env_origins[i].cpu().numpy(),'pos is',start_pose.p)
        
        # print('Searching inferencal navigation path...')
        # self.get_navigation_path()
        
        self.navigation_path = {}

        
        # if self.cfg.env.mode == 'play':
        #     for i in range(self.num_envs):
        #         for landmark in self.navigation_path[i]:
        #             pos_at_landmark = np.array([landmark[0], landmark[1], 0.0])
        #             start_pose = gymapi.Transform()
        #             start_pose.p = gymapi.Vec3(*pos_at_landmark)
        #             anymal_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, "anymal", i, self.cfg.asset.self_collisions, 0)
                
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            
            if self.cfg.env.mode == 'train':
                self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            elif self.cfg.env.mode == 'play':
                self.terrain_levels = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').reshape(self.cfg.terrain.num_rows,self.cfg.terrain.num_cols).T.reshape(-1).to(torch.long)
            
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:,:2] = self.terrain_origins[self.terrain_levels, self.terrain_types][:,:2]
            # print("LeggedRobot.terrain_origins:")
            # print(self.terrain_origins)
            # print("LeggedRobot.env_origins:")
            # print(self.env_origins)
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.
            
    # def _get_task_goals(self):
    #     """ Get goals of each environment.
    #         Same routine as self._get_env_origins
    #     """
    #     if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
    #         self.custom_origins = True
    #         self.task_goals = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
            
    #         self.terrain_levels = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').reshape(self.cfg.terrain.num_rows,self.cfg.terrain.num_cols).T.reshape(-1).to(torch.long)
            
    #         self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
    #         self.max_terrain_level = self.cfg.terrain.num_rows
    #         self.terrain_goals = torch.from_numpy(self.terrain.env_goals).to(self.device).to(torch.float)
    #         self.task_goals[:,:2] = self.terrain_goals[self.terrain_levels, self.terrain_types][:,:2]
    #     else:
    #         raise NotImplementedError("Task goals are not implemented for terrain type = "+self.cfg.terrain.mesh_type)
 
    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        self.task_ranges = class_to_dict(self.cfg.task.ranges)
        self.navi_curriculum_range = class_to_dict(self.cfg.task.curriculum_range)

        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        # if not self.terrain.cfg.measure_heights:
        #     return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        # import pdb; pdb.set_trace()
        for i in range(self.num_envs):
            starting_points = self.task_startings[i, :3]
            sphere_pose = gymapi.Transform(gymapi.Vec3(starting_points[0], starting_points[1], starting_points[2]), r=None)
            sphere_geom_start = gymutil.WireframeSphereGeometry(0.05, 8, 8, None, color=(1, 0, 0))
            gymutil.draw_lines(sphere_geom_start, self.gym, self.viewer, self.envs[i], sphere_pose)

            goal_points = self.task_goals[i, :3]
            sphere_pose = gymapi.Transform(gymapi.Vec3(goal_points[0], goal_points[1], goal_points[2]), r=None)
            sphere_geom_goal = gymutil.WireframeSphereGeometry(0.05, 8, 8, None, color=(0, 0, 1))
            gymutil.draw_lines(sphere_geom_goal, self.gym, self.viewer, self.envs[i], sphere_pose)

            origin_points = self.env_origins[i, :3]
            sphere_pose = gymapi.Transform(gymapi.Vec3(origin_points[0], origin_points[1], origin_points[2]), r=None)
            sphere_geom_origin = gymutil.WireframeSphereGeometry(0.05, 8, 8, None, color=(0, 1, 1))
            gymutil.draw_lines(sphere_geom_origin, self.gym, self.viewer, self.envs[i], sphere_pose)

            if not self.terrain.cfg.measure_heights:
                continue
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 
            
            

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")
        elif self.cfg.terrain.mesh_type == 'indoor':
            pass
            # TODO: implement a multi-layer heightfield here, according to z coordinate.
        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    # def compute_teacher_commands(self):
    #     """ Compute the best command to reach next landmark
    #     """
    #     for i in range(self.num_envs):
    #         base_pos = (self.root_states[i, :3]).cpu()
            
    #         new_poses_in_pixel = map(lambda x: self._estimate_next_pose(base_pos,x), self.command_pool)
    #         distances_in_pixel = torch.norm(torch.stack(list(new_poses_in_pixel),dim=0).cpu() - self.task_next_landmarks[i].cpu(),dim=1)

    #         randperm_indices = torch.randperm(self.command_pool.shape[0])
    #         best_command_idx = torch.argmin(distances_in_pixel[randperm_indices])
    #         best_command = torch.Tensor([*self.command_pool[randperm_indices][best_command_idx]])
    #         print(i,best_command)            
    #         self.teacher_commands[i,:3] = best_command[:]
            
    # def _estimate_next_pose(self, base_pose, command):
    #     """ Compute next pose in pixel as format torch.Tensor
        
    #     Calls estimate_next_pose() in utils.py
    #     """
    #     new_pose = estimate_next_pose(base_pose, self.cfg.locomotion.time_per_step, command)
    #     new_pose_in_pixel = self.meter_to_index(new_pose)
    #     return torch.Tensor([new_pose_in_pixel[0],new_pose_in_pixel[1]])
        
    
    #------------ reward functions----------------
    
    # def _reward_behaviour_cloning(self):
    #     # Penalize commands different from teacher_commands
    #     # command size: (num_envs, 3)
    #     return -torch.mean(
    #         torch.sqrt(
    #             torch.sum(
    #                 torch.pow(self.commands - self.teacher_commands,2.0),
    #                 dim=1)),
    #         dim=0)
    
    def _reward_success(self):
        # return self.success_buf
        return torch.norm(self.cur_pos - self.task_goals, p=2, dim = 1) < self.cfg.task.success_epsilon

    def _reward_towards(self):
        # print("************")
        # print("CUR_POSE:",self.cur_pos[0])
        # print("ROOT_STATE:", self.root_states[0,:3])
        # print("LAST_POSE:",self.last_pos[0])
        # print("TASK_GOALS:",self.task_goals[0])
        # print("RELATIVE GOAL:", self.obs_buf[0, :3])
        # print("CUR dis:", torch.norm(self.cur_pos - self.task_goals, p=2, dim = 1)[0])
        # print("LAST dis:", torch.norm(self.last_pos - self.task_goals, p=2, dim = 1)[0])
        # print("************")
        last_dis = torch.norm(self.last_pos  - self.task_goals, p=2, dim=1)
        current_dis = torch.norm(self.cur_pos - self.task_goals, p=2, dim = 1)
        # ret = 100 * (last_dis - current_dis) / last_dis
        ret = last_dis - current_dis
        # ret = 2-(torch.square(self.commands[:, 0] - 0.5) + torch.square(self.commands[:,1]) + torch.square(self.commands[:, 2]))
        if np.random.rand()<0.02:
            print(ret[0], self.commands[0])
        # print("toward reward1:", ret[0])
        return ret
    def _reward_velocity_yaw(self):
        return torch.square(self.commands[:, 2])
    def _reward_velocity_rate(self):
        # Penalize changes in velocity
        # print("************")
        # print("CUR_VEL:",self.commands[0])
        # print("LAST_VEL:",self.last_root_vel[0, :2], self.last_root_vel[0, -1])
        # # print("LAST_VEL:",self.last_commands[0])
        # print("************")
        return torch.sum(torch.square(self.last_root_vel[:,:2] - self.commands[:, :2]), dim=1) + torch.sum(torch.square(self.last_root_vel[:, -1] - self.commands[:,2]))
    def _reward_time_cost(self):
        # Penalize time cost
        return 1.0
    def _reward_imitation(self):
        ret = 2-(torch.square(self.commands[:, 0] - 0.5) + torch.square(self.commands[:,1]) + torch.square(self.commands[:, 2]))
        if np.random.rand()<0.02:
            print(ret[0], self.commands[0])
        # print("toward reward1:", ret[0])
        return ret
    # def _reward_lin_vel_z(self):
    #     # Penalize z axis base linear velocity
    #     return torch.square(self.base_lin_vel[:, 2])
    
    # def _reward_ang_vel_xy(self):
    #     # Penalize xy axes base angular velocity
    #     return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    # def _reward_orientation(self):
    #     # Penalize non flat base orientation
    #     return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    # def _reward_base_height(self):
    #     # Penalize base height away from target
    #     base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
    #     return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    # def _reward_torques(self):
    #     # Penalize torques
    #     return torch.sum(torch.square(self.torques), dim=1)

    # def _reward_dof_vel(self):
    #     # Penalize dof velocities
    #     return torch.sum(torch.square(self.dof_vel), dim=1)
    
    # def _reward_dof_acc(self):
    #     # Penalize dof accelerations
    #     return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    # def _reward_dof_pos_dif(self):
    #     return torch.sum(torch.square(self.last_dof_pos - self.dof_pos), dim=1)
    
    # def _reward_action_rate(self):
    #     # Penalize changes in actions
    #     return torch.sum(torch.square(self.last_locomotion_actions - self.locomotion_actions), dim=1)
    
    # def _reward_collision(self):
    #     # Penalize collisions on selected bodies
    #     return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1) 
        
    
    # def _reward_termination(self):
    #     # Terminal reward / penalty
    #     return self.reset_buf * ~self.success_buf
    
    # def _reward_dof_pos_limits(self):
    #     # Penalize dof positions too close to the limit
    #     out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
    #     out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
    #     return torch.sum(out_of_limits, dim=1)

    # def _reward_dof_vel_limits(self):
    #     # Penalize dof velocities too close to the limit
    #     # clip to max error = 1 rad/s per joint to avoid huge penalties
    #     return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    # def _reward_torque_limits(self):
    #     # penalize torques too close to the limit
    #     return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    # def _reward_tracking_lin_vel(self):
    #     # Tracking of linear velocity commands (xy axes)
    #     lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
    #     return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    # def _reward_tracking_ang_vel(self):
    #     # Tracking of angular velocity commands (yaw) 
    #     ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
    #     return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    # def _reward_feet_air_time(self):
    #     # Reward long steps
    #     # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
    #     contact = self.contact_forces[:, self.feet_indices, 2] > 1.
    #     contact_filt = torch.logical_or(contact, self.last_contacts) 
    #     self.last_contacts = contact
    #     first_contact = (self.feet_air_time > 0.) * contact_filt
    #     self.feet_air_time += self.dt
    #     rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
    #     rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
    #     self.feet_air_time *= ~contact_filt
    #     return rew_airTime
    
    # def _reward_stumble(self):
    #     # Penalize feet hitting vertical surfaces
    #     return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
    #          5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    # def _reward_stand_still(self):
    #     # Penalize motion at zero commands
    #     return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    # def _reward_feet_contact_forces(self):
    #     # penalize high contact forces
    #     return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)