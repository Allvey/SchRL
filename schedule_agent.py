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

import parl
import paddle
import numpy as np
import json

json_file = "config.json"

with open(json_file) as f:
    para_config = json.load(f)["para"]

# np.random.seed(para_config["seed"])


class DispatchAgent(parl.Agent):
    """Agent of Schedule-v2 env.

    Args:
        algorithm(parl.Algorithm): algorithm used to solve the problem.
    """

    def __init__(self, algorithm, act_dim, env, e_greed=0.1, e_greed_decrement=0):
        super(DispatchAgent, self).__init__(algorithm)
        assert isinstance(act_dim, int)
        self.act_dim = act_dim

        self.global_step = 0
        self.update_target_steps = 200

        self.e_greed = e_greed
        self.e_greed_decrement = e_greed_decrement

        self.adaptive_weighting = 0.7
        self.decrease_factor = 0.005

        self.env = env

        self.gain = 0

    def sample(self, obs):
        """Sample an action `for exploration` when given an observation

        Args:
            obs(np.float32): shape of (obs_dim,)

        Returns:
            act(int): action
        """
        sample = np.random.random()
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)
            # act2 = np.random.randint(10)

            # act = np.random.randint(self.act_dim)
        else:
            if np.random.random() < 0.01:
                # act = np.random.randint(self.act_dim)
                act = self.env.action_space.sample()[0]
            else:
                act = self.predict(obs)

        self.e_greed = max(0.001, self.e_greed - self.e_greed_decrement)
        return act

    def predict(self, obs):
        """Predict an action when given an observation

        Args:
            obs(np.float32): shape of (obs_dim,)

        Returns:
            act(int): action
        """
        obs = paddle.to_tensor(obs, dtype='float32')
        pred_q = self.alg.predict(obs)
        # act = pred_q.argmax().numpy()[0]
        # act1 = pred_q[0].argmax().numpy()[0]
        act = pred_q.argmax().numpy()[0]
        return act

    def learn(self, obs, act, reward, next_obs, terminal, is_sim, weight, episode=0, cof=0):
        """Update model with an episode data

        Args:
            obs(np.float32): shape of (batch_size, obs_dim)
            act(np.int32): shape of (batch_size)
            reward(np.float32): shape of (batch_size)
            next_obs(np.float32): shape of (batch_size, obs_dim)
            terminal(np.float32): shape of (batch_size)

        Returns:
            loss(float)

        """
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        act = np.expand_dims(act, axis=-1)
        reward = np.expand_dims(reward, axis=-1)
        terminal = np.expand_dims(terminal, axis=-1)
        is_sim = np.expand_dims(is_sim, axis=-1)

        obs = paddle.to_tensor(obs, dtype='float32')
        act = paddle.to_tensor(act, dtype='int32')
        reward = paddle.to_tensor(reward, dtype='float32')
        next_obs = paddle.to_tensor(next_obs, dtype='float32')
        terminal = paddle.to_tensor(terminal, dtype='float32')
        weight = paddle.to_tensor(weight, dtype='float32')
        is_sim = paddle.to_tensor(is_sim, dtype='float32')
        loss, delta, loss_t = self.alg.learn(obs, act, reward, next_obs, terminal, weight)

        # is_sim_tt = is_sim.astype(float)
        loss_tt = loss_t.numpy()
        is_sim_tt = is_sim.numpy().copy()

        sim_loss = loss_tt * is_sim_tt
        sim_loss = sum(sim_loss) / sum(is_sim_tt)

        art_loss = loss_tt * (1 - is_sim_tt)
        art_loss = sum(art_loss) / sum(1 - is_sim_tt)

        self.gain += max(0, cof * (sim_loss - art_loss))

        # with paddle.no_grad():
        #     delta += delta * is_sim * min(max(np.exp(self.gain * 0.01 - 1.5) - 1, 0), 10000)

        with paddle.no_grad():
            delta += delta * is_sim * min(max(np.exp(self.gain * 0.01 - 0.2) - 1, 0), 10000)

        # with paddle.no_grad():
        #     delta += delta * is_sim * min(max(np.exp(self.gain * 0.01 - 2) - 1, 0), 10000)

        # with paddle.no_grad():
        #     delta += delta * is_sim * min(max(self.gain * 0.01, 0), 10000)

        return loss.numpy()[0], delta, [sim_loss, art_loss]

        # return loss.numpy()[0], delta, [0, 0]