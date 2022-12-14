#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import parl


class CartpoleModel(parl.Model):
    """ Linear network to solve Cartpole problem.

    Args:
        obs_dim (int): Dimension of observation space.
        act_dim (int): Dimension of action space.
    """

    def __init__(self, obs_dim, act_dim):
        super(CartpoleModel, self).__init__()
        # hid1_size = 128
        # hid2_size = 256
        # self.fc1 = nn.Linear(obs_dim, hid1_size)
        # self.fc2 = nn.Linear(hid1_size, hid2_size)
        # self.fc3 = nn.Linear(hid2_size, act_dim)

        hid1_size = 128
        hid2_size = 256
        hid3_size = 128
        # hid4_size = 128

        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)

        self.fc3 = nn.Linear(hid2_size, hid3_size)
        # self.fc4 = nn.Linear(hid2_size, hid4_size)

        self.fc5 = nn.Linear(hid3_size, act_dim)
        # self.fc6 = nn.Linear(hid4_size, 10)

    def forward(self, obs):
        h1 = F.relu(self.fc1(obs))
        h2 = F.relu(self.fc2(h1))

        h3 = F.relu(self.fc3(h2))
        # h4 = F.relu(self.fc4(h2))

        Q1 = self.fc5(h3)
        # Q2 = self.fc6(h4)

        return Q1


# class CartpoleModel(parl.Model):
#     """ Linear network to solve Cartpole problem.
#
#     Args:
#         obs_dim (int): Dimension of observation space.
#         act_dim (int): Dimension of action space.
#     """
#
#     def __init__(self, obs_dim, act_dim):
#         super(CartpoleModel, self).__init__()
#         # hid1_size = 128
#         # hid2_size = 256
#         # self.fc1 = nn.Linear(obs_dim, hid1_size)
#         # self.fc2 = nn.Linear(hid1_size, hid2_size)
#         # self.fc3 = nn.Linear(hid2_size, act_dim)
#
#         hid1_size = 128
#         hid2_size = 256
#         hid3_size = 128
#         hid4_size = 128
#
#         self.fc1 = nn.Linear(obs_dim, hid1_size)
#         self.fc2 = nn.Linear(hid1_size, hid2_size)
#
#         self.fc3 = nn.Linear(hid2_size, hid3_size)
#         self.fc4 = nn.Linear(hid2_size, hid4_size)
#
#         self.fc5 = nn.Linear(hid3_size, act_dim)
#         self.fc6 = nn.Linear(hid4_size, 10)
#
#     def forward(self, obs):
#         h1 = F.relu(self.fc1(obs))
#         h2 = F.relu(self.fc2(h1))
#
#         h3 = F.relu(self.fc3(h2))
#         h4 = F.relu(self.fc4(h2))
#
#         Q1 = self.fc5(h3)
#         Q2 = self.fc6(h4)
#
#         return Q1, Q2
