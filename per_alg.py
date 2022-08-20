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

import copy

import numpy as np
import paddle.fluid as fluid
import paddle
import torch

import parl
from parl.core.fluid import layers
import torch.optim as optim


class PrioritizedDQN(parl.Algorithm):
    def __init__(self, model, act_dim=None, gamma=None, lr=None):
        """ DQN algorithm with prioritized experience replay.
        
        Args:
            model (parl.Model): model defining forward network of Q function
            act_dim (int): dimension of the action space
            gamma (float): discounted factor for reward computation.
            lr (float): learning rate.
        """
        self.model = model
        self.target_model = copy.deepcopy(model)

        assert isinstance(act_dim, int)
        assert isinstance(gamma, float)
        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr

    def predict(self, obs):
        """ use value model self.model to predict the action value
        """
        return self.model.value(obs)

    def learn(self, obs, action, reward, next_obs, terminal, sample_weight):
        """ update value model self.model with DQN algorithm
        """

        pred_value = self.model.value(obs)
        next_pred_value = self.target_model.value(next_obs)
        best_v = layers.reduce_max(next_pred_value, dim=1)
        best_v.stop_gradient = True
        target = reward + (
            1.0 - layers.cast(terminal, dtype='float32')) * self.gamma * best_v

        action_onehot = layers.one_hot(action, self.act_dim)
        action_onehot = layers.cast(action_onehot, dtype='float32')
        pred_action_value = layers.reduce_sum(
            action_onehot * pred_value, dim=1)
        delta = layers.abs(target - pred_action_value)
        cost = sample_weight * layers.square_error_cost(
            pred_action_value, target)
        cost = layers.reduce_mean(cost)
        optimizer = fluid.optimizer.Adam(learning_rate=self.lr, epsilon=1e-3)
        optimizer.minimize(cost)
        return cost, delta  # `delta` is the TD-error

    def sync_target(self):
        """ sync weights of self.model to self.target_model
        """
        self.model.sync_weights_to(self.target_model)


class PrioritizedDoubleDQN(parl.Algorithm):
    def __init__(self, model, act_dim=None, gamma=None, lr=None):
        """ Double DQN algorithm

        Args:
            model (parl.Model): model defining forward network of Q function.
            gamma (float): discounted factor for reward computation.
        """
        self.model = model
        self.target_model = copy.deepcopy(model)

        assert isinstance(act_dim, int)
        assert isinstance(gamma, float)

        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr

    def predict(self, obs):
        return self.model(obs)

    def learn(self, obs, action, reward, next_obs, terminal, sample_weight):
        obs = paddle.squeeze(obs)
        pred_value = self.model(obs)[0]
        action_onehot = layers.one_hot(action[:, 0], self.act_dim)
        pred_action_value = layers.reduce_sum(
            action_onehot * pred_value, dim=1)

        # calculate the target q value
        next_obs = paddle.squeeze(next_obs)
        next_action_value, _ = self.model(next_obs)
        greedy_action = layers.argmax(next_action_value, axis=-1)
        greedy_action = layers.unsqueeze(greedy_action, axes=[1])
        greedy_action_onehot = layers.one_hot(greedy_action, self.act_dim)
        next_pred_value, _ = self.target_model(next_obs)
        max_v = layers.reduce_sum(
            greedy_action_onehot * next_pred_value, dim=1, keep_dim=True)
        max_v.stop_gradient = True
        target = reward + (
            1.0 - layers.cast(terminal, dtype='float32')) * self.gamma * max_v
        target = paddle.squeeze(target)
        delta = layers.abs(target - pred_action_value)
        cost = sample_weight * layers.square_error_cost(
            pred_action_value, target)
        cost = layers.reduce_mean(cost)
        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=self.lr, epsilon=1e-3,
                                                  parameter_list=self.model.parameters())
        optimizer.minimize(cost)

        return cost, delta

    # def learn(self, obs, action, reward, next_obs, terminal, sample_weight):
    #     pred_value = self.model(obs)
    #     action_onehot = layers.one_hot(action, self.act_dim + 1)
    #     pred_action_value = layers.reduce_sum(
    #         action_onehot * pred_value, dim=1)
    #
    #     # calculate the target q value
    #     next_action_value = self.model.value(next_obs)
    #     greedy_action = layers.argmax(next_action_value, axis=-1)
    #     greedy_action = layers.unsqueeze(greedy_action, axes=[1])
    #     greedy_action_onehot = layers.one_hot(greedy_action, self.act_dim)
    #     next_pred_value = self.target_model.value(next_obs)
    #     max_v = layers.reduce_sum(
    #         greedy_action_onehot * next_pred_value, dim=1)
    #     max_v.stop_gradient = True
    #
    #     target = reward + (
    #         1.0 - layers.cast(terminal, dtype='float32')) * self.gamma * max_v
    #     delta = layers.abs(target - pred_action_value)
    #     cost = sample_weight * layers.square_error_cost(
    #         pred_action_value, target)
    #     cost = layers.reduce_mean(cost)
    #     optimizer = fluid.optimizer.Adam(learning_rate=self.lr, epsilon=1e-3)
    #     optimizer.minimize(cost)
    #     return cost, delta

    def sync_target(self):
        """ sync weights of self.model to self.target_model
        """
        self.model.sync_weights_to(self.target_model)


class DoubleDQN(parl.Algorithm):
    def __init__(self, model, act_dim=None, gamma=None, lr=None):
        """ Double DQN algorithm

        Args:
            model (parl.Model): model defining forward network of Q function.
            gamma (float): discounted factor for reward computation.
        """
        self.model = model
        self.target_model = copy.deepcopy(model)

        assert isinstance(act_dim, int)
        assert isinstance(gamma, float)

        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr

    def predict(self, obs):
        return self.model(obs)

    def learn(self, obs, action, reward, next_obs, terminal):
        obs = paddle.squeeze(obs)
        pred_value = self.model(obs)[0]
        action_onehot = layers.one_hot(action[:, 0], self.act_dim)
        pred_action_value = layers.reduce_sum(
            action_onehot * pred_value, dim=1)

        # calculate the target q value
        next_obs = paddle.squeeze(next_obs)
        next_action_value, _ = self.model(next_obs)
        greedy_action = layers.argmax(next_action_value, axis=-1)
        greedy_action = layers.unsqueeze(greedy_action, axes=[1])
        greedy_action_onehot = layers.one_hot(greedy_action, self.act_dim)
        next_pred_value, _ = self.target_model(next_obs)
        max_v = layers.reduce_sum(
            greedy_action_onehot * next_pred_value, dim=1, keep_dim=True)
        max_v.stop_gradient = True
        target = reward + (
            1.0 - layers.cast(terminal, dtype='float32')) * self.gamma * max_v
        target = paddle.squeeze(target)
        delta = layers.abs(target - pred_action_value)
        cost = layers.square_error_cost(
            pred_action_value, target)
        cost = layers.reduce_mean(cost)
        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=self.lr, epsilon=1e-3,
                                                  parameter_list=self.model.parameters())
        optimizer.minimize(cost)

        return cost, delta

    def sync_target(self):
        """ sync weights of self.model to self.target_model
        """
        self.model.sync_weights_to(self.target_model)


class DQN(parl.Algorithm):
    def __init__(self, model, gamma=None, lr=None):
        """ DQN algorithm

        Args:
            model (parl.Model): model defining forward network of Q function.
            gamma (float): discounted factor for reward computation.
            lr (float): learning rate.
        """
        self.model = model
        self.target_model = copy.deepcopy(model)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.target_model.to(device)

        assert isinstance(gamma, float)
        assert isinstance(lr, float)
        self.gamma = gamma
        self.lr = lr

        self.mse_loss = torch.nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def predict(self, obs):
        """ use value model self.model to predict the action value
        """
        with torch.no_grad():
            pred_q = self.model(obs)
        return pred_q

    def learn(self, obs, action, reward, next_obs, terminal):
        """ update value model self.model with DQN algorithm
        """
        pred_value = self.model(obs).gather(1, action)
        with torch.no_grad():
            max_v = self.target_model(next_obs).max(1, keepdim=True)[0]
            target = reward + (1 - terminal) * self.gamma * max_v
        self.optimizer.zero_grad()
        loss = self.mse_loss(pred_value, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_target(self):
        self.model.sync_weights_to(self.target_model)


class MPrioritizedDQN(parl.Algorithm):
    def __init__(self, model, gamma=None, lr=None):
        """ DQN algorithm
        Args:
            model (parl.Model): forward neural network representing the Q function.
            gamma (float): discounted factor for `accumulative` reward computation
            lr (float): learning rate.
        """
        # checks
        # check_model_method(model, 'forward', self.__class__.__name__)
        assert isinstance(gamma, float)
        assert isinstance(lr, float)

        self.model = model
        self.target_model = copy.deepcopy(model)

        self.gamma = gamma
        self.lr = lr

        self.mse_loss = paddle.nn.MSELoss(reduction='mean')
        self.optimizer = paddle.optimizer.Adam(
            learning_rate=lr, parameters=self.model.parameters())

    def predict(self, obs):
        """ use self.model (Q function) to predict the action values
        """
        return self.model(obs)

    def learn(self, obs, action, reward, next_obs, terminal, weight):
        """ update the Q function (self.model) with DQN algorithm
        """
        # Q
        pred_values = self.model(obs)
        action_dim = pred_values.shape[-1]
        action = paddle.squeeze(action, axis=-1)
        action_onehot = paddle.nn.functional.one_hot(
            action, num_classes=action_dim)
        pred_value = pred_values * action_onehot
        pred_value = paddle.sum(pred_value, axis=1, keepdim=True)

        # target Q
        with paddle.no_grad():
            max_v = self.target_model(next_obs).max(1, keepdim=True)
            target = reward + (1 - terminal) * self.gamma * max_v
        # loss = self.mse_loss(pred_value, target)

        loss = weight * layers.square_error_cost(
            pred_value, target)
        # loss = layers.square_error_cost(pred_value, target)
        loss = layers.reduce_mean(loss)

        delta = layers.abs(target - pred_value)

        # optimize
        self.optimizer.clear_grad()
        loss.backward()
        self.optimizer.step()

        return loss, delta

    def sync_target(self):
        """ assign the parameters of the training network to the target network
        """
        self.model.sync_weights_to(self.target_model)
