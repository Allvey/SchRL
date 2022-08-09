#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, softwareconsole
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import gym
import numpy as np
import parl
from parl.utils import logger, ReplayMemory

from schedule_model import CartpoleModel
from schedule_agent import DispatchAgent
from parl.algorithms.paddle import MDQN, DQN
import visdom
import time

LEARN_FREQ = 5  # training frequency
MEMORY_SIZE = 200000
MEMORY_WARMUP_SIZE = 200
# BATCH_SIZE = 64
BATCH_SIZE = 1024
LEARNING_RATE = 0.00001
# GAMMA = 0.99
GAMMA = 0.99

filename = './result-dec4.txt'


# train an episode
def run_train_episode(agent, env, rpm):
    total_mass = 0
    total_reward = 0
    obs = env.reset(train_mode=False)
    step = 0
    reward_ls = []
    while True:
        step += 1
        action = agent.sample(obs)
        # next_obs, reward, done, _, utilization, mass, weight, heu_rpm = env.step([action, 1])
        next_obs, reward, done, _, utilization, mass, weight, heu_rpm = env.step(action)

        print("agent rl action")
        print(action)

        reward_ls.append(reward)
        rpm.append(obs, action, reward, next_obs, done)

        for heu_obs, heu_action, heu_reward, heu_next_obs, heu_done in heu_rpm:
            rpm.append(heu_obs, heu_action, heu_reward,  heu_next_obs, heu_done)
            print("agent heu action")
            print(heu_action)
            print(heu_action[0])

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            # s,a,r,s',done
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample_batch(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs, batch_done)

        total_reward += reward
        obs = next_obs
        if done:
            total_mass = mass
            print(f'reward ls {reward_ls}')
            break
    return total_reward, total_mass


# evaluate 5 episodes
def run_evaluate_episodes(agent, env, eval_episodes=5, render=False):
    eval_reward = []
    eval_mass = []
    eval_utilization = []
    eval_weight = []
    for i in range(eval_episodes):
        obs = env.reset(train_mode=True)
        episode_reward = 0
        episode_mass = 0
        episode_utilization = 0
        episode_weight = 0
        while True:
            action = agent.predict(obs)
            # obs, reward, done, _, utilization, mass, weight, heu_rpm = env.step([action, 1])
            next_obs, reward, done, _, utilization, mass, weight, heu_rpm = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                episode_mass = mass
                episode_utilization = utilization
                episode_weight = weight
                break
        eval_reward.append(episode_reward)
        eval_mass.append(episode_mass)
        eval_utilization.append(episode_utilization)
        eval_weight.append(episode_weight)
    return np.mean(eval_reward), np.mean(eval_utilization), np.mean(eval_mass), np.mean(eval_weight, dtype=float)


def main():
    # env = gym.make('CartPole-v0')
    wind = visdom.Visdom(env='8-9-single')
    env = gym.make('Env-Test-v5')
    obs_dim = env.observation_space.shape[1]    # env.observation_space (1, 7)
    act_dim = env.action_space[0].n    # env.action_space ((shovels + dumps) * 2,)
    logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    # set action_shape = 0 while in discrete control environment
    # rpm = ReplayMemory(MEMORY_SIZE, obs_dim, 1)
    rpm = ReplayMemory(MEMORY_SIZE, obs_dim, 2)

    # build an agent
    model = CartpoleModel(obs_dim=obs_dim, act_dim=act_dim)
    alg = DQN(model, gamma=GAMMA, lr=LEARNING_RATE)
    agent = DispatchAgent(
        alg, act_dim=act_dim, env=env, e_greed=0.2, e_greed_decrement=1e-6)

    # # load model and evaluate
    # if os.path.exists('./model.ckpt'):
    #     agent.restore('./model.ckpt')
    #     run_evaluate_episodes(agent, env, render=False)
    #     exit()

    # warmup memory
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_train_episode(agent, env, rpm)

    max_episode = 15000

    if os.path.exists('E:/Pycharm Projects/RL/SchRL/' + filename):
        os.remove('E:/Pycharm Projects/RL/SchRL/' + filename)

    with open(filename, 'a') as file_object:
        file_object.write("{} {} {} {} {}\n".format("episode", "eval_reward", "eval_utilization", "eval_mass", "eval_weight"))

    # start training
    episode = 0
    while episode < max_episode:
        # train part
        train_total_reward = []
        train_total_mass = []
        for i in range(5):
            total_reward, total_mass = run_train_episode(agent, env, rpm)
            train_total_reward.append(total_reward)
            train_total_mass.append(total_mass)
            episode += 1

            # print("total_reward", total_reward)

        train_reward = np.mean(train_total_reward)
        train_mass = np.mean(train_total_mass)

        # test part
        eval_reward, eval_utilization, eval_mass, eval_weight = run_evaluate_episodes(agent, env, render=False)
        logger.info('episode:{}    e_greed:{}   Test reward:{}  Test mass:{} Test weight:{}'.format(
            episode, agent.e_greed, eval_reward, eval_utilization, eval_mass, eval_weight))
        with open(filename, 'a') as file_object:
            file_object.write("{} {} {} {} {}\n".format(episode, eval_reward, eval_utilization, eval_mass, eval_weight))

        wind.line([eval_mass], [episode], win='eval_mass', update='append', opts=dict(title='eval_mass'))

        wind.line([eval_reward], [episode], win='eval_reward', update='append', opts=dict(title='eval_reward'))

        wind.line([train_mass], [episode], win='train_mass', update='append', opts=dict(title='train_mass'))

        wind.line([train_reward], [episode], win='train_reward', update='append', opts=dict(title='train_reward'))
        time.sleep(0.5)

        if episode % 100 == 0:
            # save the parameters to ./model.ckpt
            save_path = './model.ckpt'
            agent.save(save_path)

    # save the parameters to ./model.ckpt
    save_path = './model.ckpt'
    agent.save(save_path)


if __name__ == '__main__':
    main()
