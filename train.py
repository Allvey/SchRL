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
from parl.algorithms.paddle import MDQN
from proportional_per import ProportionalPER
from per_alg import PrioritizedDQN, MPrioritizedDQN, MPrioritizedDoubleDQN
import visdom
import time
from tqdm import tqdm
import paddle
from collections import deque
from sklearn.linear_model import LinearRegression
import datetime
import csv

LEARN_FREQ = 5  # training frequency
MEMORY_SIZE = 2000
MEMORY_WARMUP_SIZE = 2000
# BATCH_SIZE = 64
BATCH_SIZE = 1024
LEARNING_RATE = 0.00001
# GAMMA = 0.99
GAMMA = 0.99
que = deque([])

global sim_batch


def beta_adder(init_beta, step_size=0.0001):
    beta = init_beta
    step_size = step_size

    def adder():
        nonlocal beta, step_size
        beta += step_size
        return min(beta, 1)

    return adder


def process_transitions(transitions):
    transitions = np.array(transitions)
    batch_obs = np.stack(transitions[:, 0].copy()).squeeze()
    batch_act = transitions[:, 1].copy().squeeze()
    batch_reward = transitions[:, 2].copy().squeeze()
    batch_next_obs = np.expand_dims(np.stack(transitions[:, 3]), axis=1).squeeze()
    batch_terminal = transitions[:, 4].copy().squeeze()
    batch_is_sim = transitions[:, 5].copy().squeeze()
    batch = (batch_obs, batch_act, batch_reward, batch_next_obs,
             batch_terminal, batch_is_sim)
    global sim_batch
    sim_batch = sum(batch_is_sim)
    return batch

# # train an episode
# def run_train_episode(agent, env, rpm):
#     total_mass = 0
#     total_reward = 0
#     obs = env.reset(train_mode=False)
#     step = 0
#     reward_ls = []
#     while True:
#         action = agent.sample(obs)
#         next_obs, reward, done, _, utilization, mass, weight, heu_rpm = env.step([action, 1])
#         # next_obs, reward, done, _, utilization, mass, weight, heu_rpm = env.step(action)
#
#         # reward_ls.append(reward)
#         # rpm.append(obs, action, reward, next_obs, done)
#         #
#         # step += 1
#
#         for heu_obs, heu_action, heu_reward, heu_next_obs, heu_done in heu_rpm:
#             rpm.append(heu_obs, heu_action[0], heu_reward,  heu_next_obs, heu_done)
#
#             step += 1
#
#         # train model
#         if step % LEARN_FREQ == 0:
#             # s,a,r,s',done
#             (batch_obs, batch_action, batch_reward, batch_next_obs,
#              batch_done) = rpm.sample_batch(BATCH_SIZE)
#             train_loss = agent.learn(batch_obs, batch_action, batch_reward,
#                                      batch_next_obs, batch_done)
#
#         total_reward += reward
#         obs = next_obs
#         if done:
#             total_mass = mass
#             print(f'reward ls {reward_ls}')
#             break
#     return total_reward, total_mass, step


# train an episode
def run_train_episode(agent, env, rpm, mem=None, warmup=False, episode=0, cof=0):
    total_mass = 0
    total_reward = 0
    obs = env.reset(train_mode=False)
    step = 0
    reward_ls = []
    diff = [0, 0]
    cost = 0
    while True:
        action = agent.sample(obs)
        next_obs, reward, done, _, utilization, mass, weight, heu_rpm = env.step([action, 1])

        transition = [obs, action, reward, next_obs, done, True]

        step += 1

        if warmup:
            mem.append(transition)
        else:
            rpm.store(transition)

        # for heu_obs, heu_action, heu_reward, heu_next_obs, heu_done in heu_rpm:
        #     transition = [heu_obs, heu_action[0], heu_reward,  heu_next_obs, heu_done, False]
        #     step += 1
        #
        #     if warmup:
        #         mem.append(transition)
        #     else:
        #         rpm.store(transition)

        if not warmup:
            # train model
            if step % LEARN_FREQ == 0:

                beta = get_beta()
                transitions, idxs, sample_weights = rpm.sample(beta=beta)
                batch = process_transitions(transitions)
                cost, delta, diff = agent.learn(*batch, sample_weights, episode, cof)
                rpm.update(idxs, delta)

                # is_sim = np.expand_dims(batch[-1], axis=-1)
                # is_sim = is_sim.astype(float)
                # loss_t = loss_t.numpy()
                #
                # sim_loss = loss_t * is_sim
                # sim_loss = sum(sim_loss) / sum(is_sim)
                #
                # art_loss = loss_t * (1 - is_sim)
                # art_loss = sum(art_loss) / sum(1 - is_sim)
                #
                # diff = [sim_loss, art_loss]

        total_reward += reward
        obs = next_obs
        if done:
            total_mass = mass
            print(f'reward ls {reward_ls}')
            break
    return total_reward, total_mass, step, diff, cost


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
            obs, reward, done, _, utilization, mass, weight, heu_rpm = env.step([action, 1])
            # obs, reward, done, _, utilization, mass, weight, heu_rpm = env.step(action)
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
    time_str = datetime.datetime.now().strftime('%d%m%y-%H%M%S')
    filename = f'./results/result-only-sim-{time_str}.csv'
    # env = gym.make('CartPole-v0')
    wind = visdom.Visdom(env='11-11-only-sim-1')
    import gym_sch
    # env = gym.make('sch-v0')
    env = gym.make('Env-Test-v5')
    obs_dim = env.observation_space.shape[1]    # env.observation_space (1, 7)
    act_dim = env.action_space[0].n    # env.action_space ((shovels + dumps) * 2,)
    logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    # set action_shape = 0 while in discrete control environment
    # rpm = ReplayMemory(MEMORY_SIZE, obs_dim, 0)
    rpm = ProportionalPER(alpha=0.6, seg_num=BATCH_SIZE, size=MEMORY_SIZE, framestack=1)

    # build an agent
    model = CartpoleModel(obs_dim=obs_dim, act_dim=act_dim)
    # alg = MDQN(model, gamma=GAMMA, lr=LEARNING_RATE)
    # alg = MPrioritizedDQN(model, gamma=GAMMA, lr=LEARNING_RATE)
    alg = MPrioritizedDoubleDQN(model, gamma=GAMMA, lr=LEARNING_RATE)
    agent = DispatchAgent(
        alg, act_dim=act_dim, env=env, e_greed=0.2, e_greed_decrement=1e-6)

    # # load model and evaluate
    # if os.path.exists('./model.ckpt'):
    #     agent.restore('./model.ckpt')
    #     run_evaluate_episodes(agent, env, render=False)
    #     exit()

    # # warmup memory
    # while len(rpm) < MEMORY_WARMUP_SIZE:
    #     run_train_episode(agent=agent, env=env, rpm=rpm)

    # Replay memory warmup
    total_step = 0
    with tqdm(total=MEMORY_SIZE, desc='[Replay Memory Warm Up]') as pbar:
        mem = []
        while total_step < MEMORY_WARMUP_SIZE:
            logger.info('episode:{}'.format(total_step))
            total_reward, total_mass, steps, diff, loss = run_train_episode(agent, env, rpm, mem, True)
            total_step += steps
            pbar.update(steps)
    rpm.elements.from_list(mem[:int(MEMORY_WARMUP_SIZE)])

    max_episode = 21000

    if os.path.exists('E:/Pycharm Projects/RL/SchRL/' + filename):
        os.remove('E:/Pycharm Projects/RL/SchRL/' + filename)

    with open(filename, 'a+', newline='') as file_object:
        writer = csv.writer(file_object)
        # writer.writerow("{} {} {} {}\n".format("episode", "sim_batch", "sim_loss", "art_loss"))
        header = ["episode", "eval_mass", "train_mass", "train_reward", "sim_batch", "loss",
                                         "sim_loss", "art_loss"]
        writer.writerow(header)
        # writer.writerow(
        #     "{} {} {} {} {} {}\n".format("episode", "eval_mass", "train_mass", "train_mass", "sim_batch", "loss",
        #                                  "sim_loss", "art_loss"))

    # start training
    episode = 0
    diff = [0, 0]
    loss = 0
    cof = 0

    while episode < max_episode:
        # train part
        train_total_reward = []
        train_total_mass = []

        for i in range(5):
            total_reward, total_mass, _, diff, loss = run_train_episode(agent, env, rpm, episode=episode, cof=cof)
            train_total_reward.append(total_reward)
            train_total_mass.append(total_mass)
            episode += 1

        train_reward = np.mean(train_total_reward)
        train_mass = np.mean(train_total_mass)

        que.append(train_reward)

        print("len(que)")
        print(len(que))
        if len(que) > 50:
            que.popleft()

            Y = np.expand_dims(np.array(list(que)), axis=-1)
            print("XY")
            print(Y)
            X = np.expand_dims(np.array([x for x in range(1, len(Y) + 1)]), axis=-1)
            regr = LinearRegression()
            regr.fit(X, Y)
            cof = regr.coef_[0]
            wind.line([regr.coef_[0]], [episode], win='regr.coef_', update='append', opts=dict(title='regr.coef_'))

        # test part
        eval_reward, eval_utilization, eval_mass, eval_weight = run_evaluate_episodes(agent, env, render=False)

        logger.info('episode:{}    e_greed:{}   Test reward:{}  Test mass:{} Test weight:{}'.format(
            episode, agent.e_greed, eval_reward, eval_utilization, eval_mass, eval_weight))

        global sim_batch

        # with open(filename, 'a') as file_object:
        #     file_object.write("{} {} {} {}\n".format(episode, sim_batch, diff[0], diff[1]))

        with open(filename, 'a+', newline='') as file_object:
            writer = csv.writer(file_object)
            data = [episode, eval_mass, train_mass, train_reward, sim_batch, loss,
                                             diff[0], diff[1]]
            writer.writerow(data)
            # writer.writerow(
            #     "{} {} {} {} {} {}\n".format(episode, eval_mass, train_mass, train_reward, sim_batch, loss,
            #                                  diff[0], diff[1]))

        wind.line([eval_mass], [episode], win='eval_mass', update='append', opts=dict(title='eval_mass'))

        wind.line([train_mass], [episode], win='train_mass', update='append', opts=dict(title='train_mass'))

        wind.line([train_reward], [episode], win='train_reward', update='append', opts=dict(title='train_reward'))

        wind.line([sim_batch], [episode], win='sim_batch', update='append', opts=dict(title='sim_batch'))

        wind.line([loss], [episode], win='loss', update='append', opts=dict(title='loss'))

        wind.line(
            X=np.column_stack((episode, episode)),
            Y=np.column_stack((diff[0], diff[1])),
            win='loss-multi',
            update='append',
            opts=dict(legend=["sim_loss", "art_loss"],
                      showlegend=True,
                      markers=False,
                      title='loss-multi',
                      xlabel='episodes',
                      ylabel='loss-multi',
                      fillarea=False),
        )

        print("sim_loss: ", diff[0], "art_loss: ", diff[1])

        time.sleep(0.5)

        if episode % 100 == 0:
            # save the parameters to ./model.ckpt
            save_path = f'./models/model-{time_str}.ckpt'
            agent.save(save_path)

    # save the parameters to ./model.ckpt
    save_path = f'./models/model-{time_str}.ckpt'
    agent.save(save_path)


if __name__ == '__main__':
    get_beta = beta_adder(init_beta=0.5)
    main()
