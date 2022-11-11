# #!E:\Pycharm Projects\Waytous
# # -*- coding: utf-8 -*-
# # @Time : 2022/1/23 18:49
# # @Author : Opfer
# # @Site :
# # @File : env_test.py
# # @Software: PyCharm
#
import gym
import gym_sch
import json
import numpy as np

json_file = "config.json"

with open(json_file) as f:
    para_config = json.load(f)["para"]

np.random.seed(para_config["seed"])

env = gym.make('sch-v0')

print(env.observation_space)

env.reset(False)
done_n = False
episode = 0
while not done_n:
    episode = episode + 1
    action = env.action_space.sample()
    print("action", action)
    obs_n, reward_n, done_n, info, _, _, _, _ = env.step(action)
    # print(f'episode {episode}')
    # print("action:", action)
    # print("obs:", obs_n)
    # print("reward:", reward_n)
    # print("is_done", done_n)
env.close()

import numpy as np

for _ in range(100):

    print(np.random.normal(12, 1))