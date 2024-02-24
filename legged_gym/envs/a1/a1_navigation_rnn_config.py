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
import glob

from legged_gym.envs.base.legged_robot_config import (LeggedRobotCfg,
                                                      LeggedRobotCfgPPO)

MOTION_FILES = glob.glob('datasets/mocap_motions/*')

class A1NavigationRNNCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        mode = 'train'
        num_envs = 4096#4096
        include_history_steps = None  # Number of steps of history to include.
        include_privileged_history_steps = None
        num_observations = 9
        num_privileged_obs = 9
        # num_observations = 16 + 176
        # num_privileged_obs = 16 + 176
        
        privileged_dim = 24 + 3  # privileged_obs[:,:privileged_dim] is the privileged information in privileged_obs, include 3-dim base linear vel
        height_dim = 187

        num_actions = 3 # velocity_x, velocity_y, angular_yaw,  heading (in heading mode ang_vel_yaw is recomputed from heading error)
        reference_state_initialization = False
        reference_state_initialization_prob = 0.85
        episode_length_s = 60 #[s]
        observation_with_orientatoin = True

    class locomotion:
        train_cfg_class_name = 'A1LocomotionCfgPPO'
        num_privileged_obs = None
        num_observations = 48 + 187 + 24 # amp
        num_actions = 12
        experiment_name = "a1_amp_example"
        load_run = 'layernorm'
        checkpoint = 60000
        time_per_step = 0.02
    
    class terrain( LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'
        # measure_heights = True
        measure_heights = False
        curriculum = False
        max_init_terrain_level = 5
        
        # Navigation Task: 以下参数需要改为机器人头部前方   #11 x 17
        measured_points_x = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7] # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        
        # terrain_proportions = [0.1, 0.3, 0.3, 0.1, 0.2]
        terrain_proportions = [1, 0, 0, 0, 0]
        
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0,0,0.35]# [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            #raw pos
            # 'FL_hip_joint': 0.0,  # [rad]
            # 'RL_hip_joint': 0.0,  # [rad]
            # 'FR_hip_joint': 0.0,  # [rad]
            # 'RR_hip_joint': 0.0,  # [rad]
            #
            # 'FL_thigh_joint': 0.9,  # [rad]
            # 'RL_thigh_joint': 0.9,  # [rad]
            # 'FR_thigh_joint': 0.9,  # [rad]
            # 'RR_thigh_joint': 0.9,  # [rad]
            #
            # 'FL_calf_joint': -1.8,  # [rad]
            # 'RL_calf_joint': -1.8,  # [rad]
            # 'FR_calf_joint': -1.8,  # [rad]
            # 'RR_calf_joint': -1.8,  # [rad]
            #pos2
            # 'FL_hip_joint': 0.0,   # [rad]
            # 'RL_hip_joint': 0.0,   # [rad]
            # 'FR_hip_joint': 0.0 ,  # [rad]
            # 'RR_hip_joint': 0.0,   # [rad]
            #
            # 'FL_thigh_joint': 0.8,     # [rad]
            # 'RL_thigh_joint': 0.8,   # [rad]
            # 'FR_thigh_joint': 0.8,     # [rad]
            # 'RR_thigh_joint': 0.8,   # [rad]
            #
            # 'FL_calf_joint': -1.5,   # [rad]
            # 'RL_calf_joint': -1.5,    # [rad]
            # 'FR_calf_joint': -1.5,  # [rad]
            # 'RR_calf_joint': -1.5,    # [rad]
            #pos3
            # 'FL_hip_joint': 0.0,  # [rad]
            # 'RL_hip_joint': 0.0,  # [rad]
            # 'FR_hip_joint': 0.0,  # [rad]
            # 'RR_hip_joint': 0.0,  # [rad]
            #
            # 'FL_thigh_joint': 0.7,  # [rad]
            # 'RL_thigh_joint': 0.7,  # [rad]
            # 'FR_thigh_joint': 0.7,  # [rad]
            # 'RR_thigh_joint': 0.7,  # [rad]
            #
            # 'FL_calf_joint': -1.4,  # [rad]
            # 'RL_calf_joint': -1.4,  # [rad]
            # 'FR_calf_joint': -1.4,  # [rad]
            # 'RR_calf_joint': -1.4,  # [rad]
            #pos4
            'FL_hip_joint': 0.0,  # [rad]
            'RL_hip_joint': 0.0,  # [rad]
            'FR_hip_joint': 0.0,  # [rad]
            'RR_hip_joint': 0.0,  # [rad]

            'FL_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 1.0,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 1.0,  # [rad]

            'FL_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]
        }
    class sim:
        dt =  0.005
        substeps = 1
        gravity = [0., 0. ,-9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23 #2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2 # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 28.}  # [N*m/rad]
        damping = {'joint': 0.7}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    # class terrain( LeggedRobotCfg.terrain ):
    #     mesh_type = 'plane'
    #     measure_heights = False

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf"]
        # terminate_after_contacts_on = [
        #     "base", "FL_calf", "FR_calf", "RL_calf", "RR_calf",
        #     "FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"]
        # self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter

    class domain_rand:
        randomize_friction = True
        friction_range = [0.25, 1.75]
        randomize_base_mass = True
        added_mass_range = [-1., 1.]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0
        randomize_gains = True
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]

    class noise:
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            # dof_pos = 0.03
            # dof_vel = 1.5
            # lin_vel = 0.1
            # ang_vel = 0.3
            # gravity = 0.05
            # height_measurements = 0.1

            base_pos = 0.05
            dof_pos = 0.01
            dof_vel = 0.05
            lin_vel = 0.1
            ang_vel = 0.05
            gravity = 0.02
            quat = 0.01
            height_measurements = 0.1

    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        reward_curriculum = False
        # class scales( LeggedRobotCfg.rewards.scales ):
            # termination = 0.0
            # tracking_lin_vel = 1.5 * 1. / (.005 * 6)
            # tracking_ang_vel = 0.5 * 1. / (.005 * 6)
            # lin_vel_z = 0.0
            # ang_vel_xy = 0.0
            # orientation = 0.0
            # torques = 0.0
            # dof_vel = 0.0
            # dof_acc = 0.0
            # base_height = 0.0
            # feet_air_time =  0.0
            # collision = 0.0
            # feet_stumble = 0.0
            # action_rate = 0.0
            # stand_still = 0.0
            # dof_pos_limits = 0.0

            # # termination = -0.0
            # tracking_lin_vel = 1.0
            # tracking_ang_vel = 0.5
            # lin_vel_z = 0#-2.0
            # ang_vel_xy = 0#-0.05
            # orientation = 0#-2.0
            # torques = -0.0001
            # dof_vel = -0.
            # dof_acc = -2.5e-7
            # base_height = -0.
            # feet_air_time =  1.0
            # collision = -0.1
            # feet_stumble = -0.0
            # action_rate = -0.01
            # # dof_pos_dif = -0.1
            # # stand_still = -0.
            
        class scales:
            # behaviour_cloning = 1.0
            # success = 100
            # collision = -10
            # velocity_rate = 0
            towards = 5
            # time_cost = -0.1
            # imitation = 5.0
            # termination = -30
            # velocity_yaw = - 10

        only_positive_rewards = False


    

    class normalization( LeggedRobotCfg.normalization ):
        clip_actions = 6
        clip_commands = 0.6
    
    class task: # Regular Room Task
        class ranges:
            # Scale: m
            # From env center to starting point
            starting_x = [-1.5, 1.5]
            starting_y = [-1.5, 1.5]
            # Heading on starting point
            starting_yaw = [-3.1415, 3.1415]
            # From env center to goal
            goal_x = [-1.5, 1.5]
            goal_y = [-1.5, 1.5]
            
        robot_collision_box = (0.5,0.5)
        min_path_length = 5 # Scale: pixels
        show_checking = False

        # curriculum on navigation
        # curriculum = True
        curriculum = False
        success_epsilon = 0.4 # [m] (Original: 2)

        class curriculum_range:
            max_starting_xy_curriculum = 10.0 # [m]
            max_goal_xy_curriculum = 10.0 # [m]
            min_success_epsilon = 0.3 # [m] (Finally: 0.3) 


    
    class commands:
        # curriculum = False
        # max_curriculum = 1.
        '''If use tanh, set the noise_std as 0.25, otherwise, the output will be multiple the commands_scale(0.25) after clipping -4,4'''
        '''When use tanh, need to set the commands_scale = 1.'''
        # commands_scale = 0.25 # The actually commands is the output of policy * commands_scale
        commands_scale = 1
        curriculum = False
        max_lin_vel_x_curriculum = 1.
        max_lin_vel_y_curriculum = 1.
        max_ang_vel_yaw_curriculum = 1.0
        num_commands = 3 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        # resampling_time = 10. # time before command are changed[s] NO USE
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            # lin_vel_x = [-1.0, 2.0] # min max [m/s]
            # lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            # ang_vel_yaw = [-1.57, 1.57]    # min max [rad/s]
            # heading = [-3.14, 3.14]
            lin_vel_x = [-1., 1.] # min max [m/s]
            # lin_vel_x = [0.5, 0.5]
            lin_vel_y = [-1., 1.]   # min max [m/s]
            ang_vel_yaw = [-1., 1.]
            # ang_vel_yaw = [0, 0]    # min max [rad/s]
            heading = [-3.14/4, 3.14/4]
        class choices:
            lin_vel_x = [0.4]
            lin_vel_y = [0.0]
            ang_vel_yaw = [-0.4, -0.3,-0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4]

class A1NavigationCfgRNNPPO( LeggedRobotCfgPPO ):
    runner_class_name = 'OnPolicyRunner'
    
    class policy:         
        # init_noise_std = 0.7
        # include_history_steps = 5
        include_history_steps = None
        MLP_hidden_dims = [512, 512]
        latent_dim = 512
        activation = 'relu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

        '''If use_tanh, set the noise_std as 0.25, otherwise, the output will be multiple the commands_scale(0.25) after clipping -4,4'''
        # use_tanh = False
        # init_noise_std = 1.0

        use_tanh = True
        # init_noise_std = 0.25

        rnn_num_layers = 1
        rnn_hidden_size = 512
        rnn_type = 'gru'

    class algorithm:
        # training params
        value_loss_coef = 1.0
        # use_clipped_value_loss = True
        use_clipped_value_loss = False
        clip_param = 0.2
        # entropy_coef = 1e-3
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 5.e-4 #5.e-4
        schedule = 'adaptive' # could be adaptive, fixed
        gamma = 0.9
        # gamma = 0.5
        lam = 0.95
        desired_kl = 1e-2
        max_grad_norm = 0.5

    class runner( LeggedRobotCfgPPO.runner ):

        # logging
        save_interval = 200 # check for potential saves every this many iterations
        experiment_name = 'a1_navigation_test'
        run_name = 'PPO_NAV_TEST'
        
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
        
        policy_class_name = 'ActorCriticRecurrent'
        # policy_class_name = "ActorCritic"
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 10000 # number of policy updates

class A1LocomotionCfgPPO( LeggedRobotCfgPPO ):
    runner_class_name = 'AMPTrainTeacherRunner'
    
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        include_history_steps = None  # Number of steps of history to include.
        num_observations = 48 + 187 + 24 #+ 16 #+ 11 * 17
        num_privileged_obs = 48 + 187 + 24 #+ 16 #+ 11 * 17
        privileged_dim = 24 + 3  # privileged_obs[:,:privileged_dim] is the privileged information in privileged_obs, include 3-dim base linear vel
        # privileged_dim = 0
        height_dim = 187  # privileged_obs[:,-height_dim:] is the heightmap in privileged_obs
        reference_state_initialization = False
        reference_state_initialization_prob = 0.85
        amp_motion_files = MOTION_FILES
        history_length = 5
        num_actions = 12

    class policy:
        init_noise_std = 1.0
        encoder_hidden_dims = [256, 128]
        predictor_hidden_dims = [64, 32]
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [512, 256, 128]
        latent_dim = 32 + 3
        # height_latent_dim = 16  # the encoder in teacher policy encodes the heightmap into a height_latent_dim vector
        # privileged_latent_dim = 8  # the encoder in teacher policy encodes the privileged infomation into a privileged_latent_dim vector
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        # rnn_type = 'lstm'
        # rnn_hidden_size = 512
        # rnn_num_layers = 1

    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        vel_predict_coef = 1.0
        amp_replay_buffer_size = 1000000
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'flat_push1'
        experiment_name = 'a1_amp_example'
        algorithm_class_name = 'AMPPPO'
        policy_class_name = 'TeacherActorCritic'
        max_iterations = 20000 # number of policy updates
        save_interval = 1000

        amp_reward_coef = 0.5 * 0.02  #set to 0 means not use amp reward
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 2000000
        amp_task_reward_lerp = 0.3
        amp_discr_hidden_dims = [1024, 512]

        min_normalized_std = [0.05, 0.02, 0.05] * 4

  
