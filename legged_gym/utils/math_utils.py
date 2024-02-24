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

import torch
from torch import Tensor
import numpy as np
from isaacgym.torch_utils import quat_apply, normalize, get_euler_xyz
from typing import Tuple

# @ torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

# @ torch.jit.script
def wrap_to_pi(angles):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles

# @ torch.jit.script
def torch_rand_sqrt_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    r = 2*torch.rand(*shape, device=device) - 1
    r = torch.where(r<0., -torch.sqrt(-r), torch.sqrt(r))
    r =  (r + 1.) / 2.
    return (upper - lower) * r + lower

def estimate_next_pose(pos,dt,command):
    # type: (torch.Tensor, float, Tuple) -> torch.Tensor
    # Estimate new pose by regarding arc length = v*dt
    vx, vy, w = command
    new_pos = torch.clone(pos)
    new_pos[0] = pos[0] + vx * dt * np.cos(pos[2])
    new_pos[1] = pos[1] + vx * dt * np.sin(pos[2])
    new_pos[2] = pos[2] + w *dt
    return new_pos

def coordinate_transform(new_origin_in_old, target_point_in_old):
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
    # Transform new coordinate to target coordinate
    # new_coordinate_in_origin: (x, y, theta): coordinate of new origin in old coordinate
    # target_in_origin: (x, y): coordiante of target point in old coordinate
    # return: target_in_new: (x, y): coordinate of target point in new coordinate
    dx,dy,dtheta = new_origin_in_old
    cos_ = torch.cos(dtheta)
    sin_ = torch.sin(dtheta)
    transform_matrix = torch.Tensor([
        [cos_, sin_, -cos_*dx-sin_*dy],
        [-sin_, cos_, sin_*dx-cos_*dy],
        [0.0, 0.0, 1.0]
        ])
    _target_point_in_old = torch.Tensor([target_point_in_old[0], target_point_in_old[1], 1.0])
    _target_point_in_new = torch.matmul(transform_matrix, _target_point_in_old)
    
    return _target_point_in_new[:2]

def coordinate_transform_3D(quat, root, target_point_in_global):
    '''
    MARK: THIS IMPLEMENT IS NOT UNDER TEST! HAVE A LOT OF PROBABILITY TO BE WRONG!!
    type: (torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
    Transform the target_point_in_global to target_point_in_local with 3D representation.
    The local coordinate is defined by the quaternion (x, y, z, w).
    The positions of robots are defined by root (x, y, z) in global coordinate.
    quat: Tensor of shape (N, 4)
    root: Tensor of shape (N, 3)
    target_point_in_global: Tensor of shape (N, 3)
    N is the number of parrallel environments.
    returns the target_point_in_local: Tensor of shape (N, 3) in which coordinates are computed by the quats and roots.
    '''
    # roll, pitch, yaw = get_euler_xyz(quat)
    # Transition_Matrixs = torch.zeros((quat.shape[0], 3, 3))
    # Transition_Matrixs[:, 0, 0] = torch.cos(yaw)*torch.cos(pitch)
    # Transition_Matrixs[:, 0, 1] = torch.cos(yaw)*torch.sin(pitch)*torch.sin(roll)-torch.sin(yaw)*torch.cos(roll)
    # Transition_Matrixs[:, 0, 2] = torch.cos(yaw)*torch.sin(pitch)*torch.cos(roll)+torch.sin(yaw)*torch.sin(roll)
    # Transition_Matrixs[:, 1, 0] = torch.sin(yaw)*torch.cos(pitch)
    # Transition_Matrixs[:, 1, 1] = torch.sin(yaw)*torch.sin(pitch)*torch.sin(roll)+torch.cos(yaw)*torch.cos(roll)
    # Transition_Matrixs[:, 1, 2] = torch.sin(yaw)*torch.sin(pitch)*torch.cos(roll)-torch.cos(yaw)*torch.sin(roll)
    # Transition_Matrixs[:, 2, 0] = -torch.sin(pitch)
    # Transition_Matrixs[:, 2, 1] = torch.cos(pitch)*torch.sin(roll)
    # Transition_Matrixs[:, 2, 2] = torch.cos(pitch)*torch.cos(roll)
    # Transition_Matrixs = Transition_Matrixs.to(quat.device)
    # target_point_in_global = target_point_in_global - root
    # target_point_in_local = torch.matmul(Transition_Matrixs, target_point_in_global.unsqueeze(-1)).squeeze(-1)
    # return target_point_in_local
    device = root.device
    relative_points = (target_point_in_global - root).detach().clone().to(device)
    from scipy.spatial.transform import Rotation as R
    rotation_matrices = torch.tensor(R.from_quat(quat.detach().cpu().numpy()).as_matrix()).to(device).float()
    # print(rotation_matrices.shape)
    # print(relative_points.shape)
    transformed_points = torch.bmm(rotation_matrices, relative_points.unsqueeze(-1)).squeeze(-1)
    return transformed_points
                                                                                      

    
