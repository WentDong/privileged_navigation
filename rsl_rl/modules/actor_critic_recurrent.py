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

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from .actor_critic import ActorCritic, get_activation
from rsl_rl.utils import unpad_trajectories

class ActorCriticRecurrent(nn.Module):
    is_recurrent = True
    def __init__(self,  num_obs,
                        num_critic_obs,
                        num_actions,
                        MLP_hidden_dims=[512, 512],
                        latent_dim = 512,
                        activation='elu',
                        rnn_type='lstm',
                        rnn_hidden_size=512,
                        rnn_num_layers=1,
                        use_tanh = False,
                        **kwargs):
        super(ActorCriticRecurrent, self).__init__()
        if kwargs:
            print("ActorCriticRecurrent.__init__ got unexpected arguments, which will be ignored: " + str(kwargs.keys()),)

        activation = get_activation(activation)
        MLP_Encoder = []
        MLP_Encoder.append(nn.Linear(num_obs, MLP_hidden_dims[0]))
        MLP_Encoder.append(activation)
        for l in range(len(MLP_hidden_dims)):
            if  l == len(MLP_hidden_dims) - 1:
                MLP_Encoder.append(nn.Linear(MLP_hidden_dims[l], latent_dim))
            else:
                MLP_Encoder.append(nn.Linear(MLP_hidden_dims[l], MLP_hidden_dims[l+1]))
                MLP_Encoder.append(activation)
        self.MLP_Encoder = nn.Sequential(*MLP_Encoder)

        self.memory = Memory(latent_dim, type=rnn_type, num_layers=rnn_num_layers, hidden_size=rnn_hidden_size)
        actor = []
        actor.append(nn.Linear(rnn_hidden_size, num_actions))
        if use_tanh:
            actor.append(nn.Tanh())
        self.actor = nn.Sequential(*actor)
        self.critic = nn.Linear(rnn_hidden_size, 1)
        self.std = nn.Linear(rnn_hidden_size, num_actions)

        self.distribution = None

    def reset(self, dones=None):
        self.memory.reset(dones)
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations, masks=None, hidden_states=None):
        hid = self.MLP_Encoder(observations)
        rnn_hid = self.memory(hid, masks, hidden_states).squeeze(0)
        mean = self.actor(rnn_hid)
        std = torch.clamp(self.std(rnn_hid), 1e-6, 1.0)
        self.distribution = Normal(mean, mean * 0. + std)

    def act(self, observations, masks=None, hidden_states=None):
        self.update_distribution(observations, masks, hidden_states)
        return self.distribution.sample()

    def act_inference(self, observations):
        hid = self.MLP_Encoder(observations)
        rnn_hid = self.memory(hid).squeeze(0)
        action = self.actor(rnn_hid)
        return action

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        hid = self.MLP_Encoder(critic_observations)
        input_c = self.memory(hid, masks, hidden_states).squeeze(0)
        return self.critic(input_c)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_hidden_states(self):
        return self.memory.hidden_states

class Memory(torch.nn.Module):
    def __init__(self, input_size, type='lstm', num_layers=1, hidden_size=256):
        super().__init__()
        # RNN
        rnn_cls = nn.GRU if type.lower() == 'gru' else nn.LSTM
        self.rnn = rnn_cls(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.hidden_states = None
    
    def forward(self, input, masks=None, hidden_states=None):
        batch_mode = masks is not None
        if batch_mode:
            # batch mode (policy update): need saved hidden states
            if hidden_states is None:
                raise ValueError("Hidden states not passed to memory module during policy update")
            out, _ = self.rnn(input, hidden_states)
            out = unpad_trajectories(out, masks)
        else:
            # inference mode (collection): use hidden states of last step
            out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out

    def reset(self, dones=None):
        # When the RNN is an LSTM, self.hidden_states_a is a list with hidden_state and cell_state
        for hidden_state in self.hidden_states:
            hidden_state[..., dones, :] = 0.0