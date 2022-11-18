# # #!E:\Pycharm Projects\Waytous
# # # -*- coding: utf-8 -*-
# # # @Time : 2022/1/23 18:49
# # # @Author : Opfer
# # # @Site :
# # # @File : env_test.py
# # # @Software: PyCharm
# #
# import gym
# import gym_sch
# import json
# import numpy as np
#
# print(gym.__version__)
#
# json_file = "config.json"
#
# with open(json_file) as f:
#     para_config = json.load(f)["para"]
#
# np.random.seed(para_config["seed"])
#
# # env = gym.make('sch-v0')
#
# env = gym.make('Env-Test-v5')
#
# print(env.spec.max_episode_steps)
#
# env.reset(False)
# done_n = False
# episode = 0
# while not done_n:
#     episode = episode + 1
#     action = env.action_space.sample()
#     print("action", action)
#     obs_n, reward_n, done_n, info, _, _, _, _ = env.step(action)
#     # print(f'episode {episode}')
#     # print("action:", action)
#     # print("obs:", obs_n)
#     # print("reward:", reward_n)
#     # print("is_done", done_n)
# env.close()

#!E:\Pycharm Projects\Waytous
# -*- coding: utf-8 -*-
# @Time : 2021/5/6 11:26
# @Author : Opfer
# @Site :
# @File : sim_py.py
# @Software: PyCharm


import simpy
import numpy as np
import sys
import copy
from typing import List

# np.random.seed(10)

sys.path.append(r'D:/SoftWares/Anaconda3/envs/RL/Lib/site-packages/gym/envs/user')  # 要用绝对路径

global go_to_unload_point_vehical_num  # 用于保存每条卸载道路上经过的车辆个数
global go_to_excavator_vehical_num  # 用于保存每条装载道路上的经过的车辆总个数
global sim_start_time  # 用于保存仿真程序运行的真实起始时间

global walking_go_to_unload_point_vehical_num  # 卸载道路上正在行走的车辆个数
global walking_go_to_excavator_vehical_num  # 装载道路上正在行走的车辆个数

global reach_excavator_time_list  # 驶往挖机矿卡预期抵达时间
global reach_dump_time_list  # 驶往卸点预期抵达时间

global loading_in_excavator_vehical_num  # 挖机装车数
global unloading_in_unload_point_vehical_num  # 卸点卸车数

global wating_in_excavator_vehical_num  # 挖机排队等待装载矿卡数
global wating_in_unload_point_vehical_num  # 卸点排队等待卸载矿卡数

global excavator_loaded_vehicle_num  # 挖机完成装载的车次数
global dump_loaded_vehicle_num  # 卸点完成卸载的车次数

global waiting_vehicle_loaded_time  # 完成所有排队等待车辆装载时间
global waiting_vehicle_unloaded_time  #  完成所有排队等待车辆卸载时间

global dump_available_time  # 卸点最近一次可用时间
global shovel_available_time  # 挖机最近一次可用时间
global truck_available_time  # 矿卡最近一次可用时间

global real_shovel_mass  # 挖机实际装载量
global real_dump_mass  # 卸点实际卸载量

global last_real_shovel_mass  # 上一次调度任务挖机实际装载量
global last_real_dump_mass  # 上一次调度任务卸点实际卸载量

global recording  # 数据统计{时间(min), 产量(tonnes), 油耗(liters)}

global truck_location  # 矿卡当前位置

global next_dest  # 矿卡下一目的地

global action_time  # 调度下发时间

global truck_waiting  # 车辆等待时间
global truck_waiting_from_last_act  # 车辆自上一次动作到当前动作等待时间

global request_id  # 请求编号 1-请求调度, 2-行走结束

global goto_excavator_traffic_flow_num  # 驶往挖机各运输路线实际车次
global goto_dump_traffic_flow_num  # 驶往卸点各运输路线实际车次

global load_start_time  # 挖机开始装载时间
global unload_start_time  # 卸点开始卸载时间

global load_end_time  # 挖机结束装载时间
global unload_end_time  # 卸点结束装载时间

global mass_list  # 统计每个班次产量

class global_var:
    # 班次时间(min)
    T = 480
    # 矿卡载重(吨)payload = 220
    payload = 220
    # 矿卡数量
    truck_num = 30
    # # 电铲数量
    # n = 4
    # # 卸点数量
    # m = 4
    # # 电铲数量
    n = 3
    # 卸点数量
    m = 2
    # 电铲数量
    # n = 6
    # # 卸点数量
    # m = 4
    # 矿卡平均行驶速度
    empty_speed = 25
    heavy_speed = 22

    # # # 各路线距离
    # dis =[[4.49, 1.91, 7.49, 1.44, 1.64, 7.43, 7.03, 8.23, 6.63, 3.17],
    #      [0.95, 9.63, 3.86, 8.92, 4.03, 7.62, 4.02, 2.12, 0.02, 0.27],
    #      [5.11, 8.38, 6.37, 0.46, 5.83, 9.05, 3.95, 4.02, 9.48, 4.95],
    #      [6.83, 6.41, 2.78, 0.08, 6.43, 9.76, 4.37, 0.32, 9.98, 3.19],
    #      [5.68, 5.47, 0.96, 8.25, 2.87, 4.27, 0.87, 7.54, 2.67, 8.27],
    #      [6.81, 8.2 , 1.3 , 2.54, 8.18, 2.23, 4.34, 7.16, 3.75, 2.35],
    #      [7.34, 4.73, 7.8 , 9.78, 4.89, 0.26, 8.56, 3.79, 3.9 , 6.19],
    #      [3.97, 5.48, 1.99, 4.11, 1.15, 0.55, 0.64, 4.96, 0.53, 5.63],
    #      [0.34, 4.25, 0.56, 1.35, 1.62, 0.47, 4.03, 8.99, 1.23, 5.22],
    #      [5.64, 8.98, 8.26, 9.87, 6.59, 7.89, 8.43, 5.42, 3.4 , 5.39]]

    # # 各路线距离
    # dis =[[4.49, 1.91, 7.49, 1.44, 1.64, 7.43, 7.03, 8.27],
    #      [0.95, 9.63, 3.86, 8.92, 4.03, 7.62, 4.02, 2.12],
    #      [5.11, 8.38, 6.37, 0.46, 5.83, 9.05, 3.95, 4.02],
    #      [6.83, 6.41, 2.78, 0.08, 6.43, 9.76, 4.37, 0.32],
    #      [5.68, 5.47, 0.96, 8.25, 2.87, 4.27, 0.87, 7.54],
    #      [6.81, 8.2 , 1.3 , 2.54, 8.18, 2.23, 4.34, 7.16]]

    # dis =[[6.3,  12.16,  8.79,  9.43, 14.79,  7.96],
    #       [5.87,  8.71, 10.92,  4.67, 12.85,  12.37],
    #       [1.93, 14.76,  2.25,  3.19,  6.45,  11.6],
    #       [5.04,  7.79,  0.05,  1.63,  1.07,  7.27]]


    # dis = [[4.01, 4.45, 5.32, 6.31],
    #        [5.45, 3.65, 4.75, 4.26],
    #        [4.481, 4.481, 5.481, 5.39],
    #        [6.481, 3.481, 7.481, 4.81]]

    # 各路线距离
    # dis = [[4.01,   10.45,     5.32,    26.31],
    #         [5.45,  3.65,     12.75,    4.26],
    #         [4.481, 20.481,    5.481,   5.39],
    #         [36.481, 10.481,    30.481,   4.81]]

    # dis = [[4.01,   10.45,     5.32,    26.31],
    #         [5.45,  3.65,     12.75,    4.26],
    #         [4.481, 100,    5.481,   5.39],
    #         [100, 10.481,    100,   4.81]]

    # dis = [[4.01,   10.45]]

    # dis = [[4.01, 1.45]]

    # dis = [[4.01,   10.45,     5.32],
    #         [15.45,  3.65,     12.75]]

    dis = [[4.01,   10.45,     15.32],
            [15.45,  3.65,     12.75]]

    # # 各路线距离
    # dis = [[4.01]]

    # 各路线空载行驶时间（min）
    com_time = 60 * np.array(dis) / empty_speed
    go_time = 60 * np.array(dis) / heavy_speed

    # 电铲装载速度&卸点卸载速度（吨/时）

    # load_capacity = np.array([1600, 2000, 2000, 1600, 2000, 2000, 1600, 2000, 2000, 1600])
    # unload_capacity = np.array([2375, 2375, 2375, 2375, 2375, 2375, 2375, 2375, 2375, 2375])

    # load_capacity = np.array([1600, 2000, 2000, 1600, 2000, 2000, 1600, 2000])
    # unload_capacity = np.array([2375, 2375, 2375, 2375, 2375, 2375])

    # load_capacity = np.array([1600, 2000, 2000, 1600, 2000, 2000])
    # unload_capacity = np.array([2375, 2375, 2375, 2375])

    # load_capacity = np.array([1600, 2000, 2000]);
    # load_capacity = np.array([1600, 2000, 2000, 1600])
    # unload_capacity = np.array([2375, 2375, 2375, 2375])

    # load_capacity = np.array([1600, 2000]);
    # unload_capacity = np.array([2375]);

    load_capacity = np.array([1600, 2000, 2000])
    unload_capacity = np.array([2375, 2375])

    # 电铲装载时间&卸点卸载时间（min）
    loading_time = np.round(60 * (payload) / load_capacity, 3)
    unloading_time = np.round(60 * (payload) / unload_capacity, 3)

    # 装载及卸载时间维度扩展
    loading_time_dims = np.expand_dims(loading_time, 0).repeat(m, axis=0)
    unloading_time_dims = np.expand_dims(unloading_time, 1).repeat(n, axis=1)

    # 矿卡空载行驶能耗
    empty_power = 85
    # 矿卡重载行驶能耗
    heavy_power = 150
    # 矿卡空转能耗
    idle_power = 40
    # 电铲闲置能耗
    # shovel_idle_power = [6.6, 9.5, 9.5, 6.6, 9.5, 9.5, 6.6, 9.5, 9.5, 6.6]
    # shovel_work_power = [117, 130, 130, 117, 130, 130, 117, 130, 130, 117]

    # shovel_idle_power = [6.6, 9.5, 9.5, 6.6, 9.5, 9.5, 6.6, 9.5]
    # shovel_work_power = [117, 130, 130, 117, 130, 130, 117, 130]

    # shovel_idle_power = [6.6, 9.5, 9.5, 6.6, 9.5, 9.5]
    # shovel_work_power = [117, 130, 130, 117, 130, 130]

    shovel_idle_power = [6.6, 9.5, 9.5]
    shovel_work_power = [117, 130, 130]

    # shovel_idle_power = [6.6, 9.5, 9.5, 6.6]
    # shovel_work_power = [117, 130, 130, 117]

    # shovel_idle_power = [6.6, 9.5];
    # shovel_work_power = [117, 130];

    # shovel_idle_power = [6.6];
    # shovel_work_power = [117];

    # shovel_work_power = np.divide(shovel_work_power, 60)

    # 速度油耗关系(速度:km/h, 节油系数:%)
    fuel_speed_empty = [[22.5, 0.2], [23, 0.18], [23.5, 0.17], [24, 0.15], [24.5, 0.11], [25, 0.0]]
    fuel_speed_heavy = [[19.5, 0.2], [20, 0.18], [20.5, 0.17], [21, 0.15], [21.5, 0.11], [22, 0.0]]

    # 各挖机/卸点目标产量
    # dump_target_mass = np.array([15000, 15000, 15000, 15000])
    # shovel_target_mass = np.array([15000, 15000, 15000, 15000])

    # dump_target_mass = np.array([15000])
    # shovel_target_mass = np.array([15000, 15000])

    shovel_target_mass = np.array([15000, 15000, 15000])
    dump_target_mass = np.array([15000, 15000])

    # dump_target_mass = np.array([15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000])
    # shovel_target_mass = np.array([15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000])

    # shovel_target_mass = np.array([15000, 15000, 15000, 15000, 15000, 15000, 15000, 15000])
    # dump_target_mass = np.array([15000, 15000, 15000, 15000, 15000, 15000])

    # shovel_target_mass = np.array([15000, 15000, 15000, 15000, 15000, 15000])
    # dump_target_mass = np.array([15000, 15000, 15000, 15000])


    #
    # dump_target_mass = np.array([15000])
    # shovel_target_mass = np.array([15000])
    # shovel_target_mass = np.array([15000, 15000, 15000])


# 对于每个全局变量，都需要定义get_value和set_value接口
def set_para(name):
    global_var.name = name


def get_para(name):
    if name == 'T&p':
        return global_var.T, global_var.payload
    elif name == 'mnt':
        return global_var.m, global_var.n, global_var.truck_num
    elif name == 'time':
        return global_var.com_time, global_var.go_time, global_var.loading_time, global_var.unloading_time
    elif name == 'energy':
        return global_var.empty_power, global_var.heavy_power, global_var.idle_power, \
               global_var.shovel_idle_power, global_var.shovel_work_power
    elif name == 'dis':
        return global_var.dis
    elif name == 'fuel_speed':
        return global_var.fuel_speed_empty, global_var.fuel_speed_heavy
    elif name == 'road':
        return global_var.road_ava
    elif name == 'capacity':
        return global_var.load_capacity, global_var.unload_capacity
    elif name == 'target':
        return global_var.shovel_target_mass, global_var.dump_target_mass
    elif name == 'speed':
        return global_var.empty_speed, global_var.heavy_speed


##############################
#  仿真参数配置，整个仿真的基本单位是分钟
##############################
T, payload = get_para("T&p")
dumps, shovels, truck_num = get_para("mnt")
com_time, go_time, loading_time, unloading_time = get_para("time")
dis = get_para("dis")
fuel_speed_empty, fuel_speed_heavy = get_para("fuel_speed")
(
    empty_power,
    heavy_power,
    idle_power,
    shovel_idle_power,
    shovel_work_power,
) = get_para("energy")
shovel_target_mass, dump_target_mass = get_para("target")
empty_speed, heavy_speed = get_para("speed")


def truck_schedule_send_post_start(truck_id: int, start_area: int, task_id: int, env: object):
    global shovel_available_time
    global truck_available_time
    global real_shovel_mass
    global last_real_shovel_mass
    target = np.argmax(
        1000 * (1 - real_shovel_mass / shovel_target_mass)
        / (np.maximum(shovel_available_time, env.now + com_time[start_area][:]) - env.now))
    shovel_available_time[target] = max(shovel_available_time[target], truck_available_time[truck_id] + com_time[
        start_area][target])
    truck_available_time[truck_id] = shovel_available_time[target]
    last_real_shovel_mass = copy.deepcopy(real_shovel_mass)
    real_shovel_mass[target] += payload
    return target


# def truck_schedule_send_post(truck_id, start_area, task_id):
#     global truck_available_time
#     global shovel_available_time
#     global last_real_shovel_mass
#     global real_shovel_mass
#     global dump_available_time
#     global last_real_dump_mass
#     global real_dump_mass
#
#     # if task_id == 2:
#     #     return next_dest[0, 0]
#     # elif task_id == 3:
#     #     return next_dest[0, 1]
#
#     # # return 0
#     # if task_id == 2:    # 代表从卸载点到装载点
#     #     return np.random.randint(0, shovels, 1)[0]
#     #
#     # if task_id == 3:    # 代表从装载点到卸载点
#     #     return np.random.randint(0, dumps, 1)[0]
#
#     # return 0
#     if task_id == 2:    # 代表从卸载点到装载点
#         if next_dest >= shovels:
#             return np.random.randint(0, shovels, 1)[0]
#         else:
#             return next_dest
#
#     if task_id == 3:    # 代表从装载点到卸载点
#         if next_dest < shovels:
#             return np.random.randint(0, dumps, 1)[0]
#         else:
#             return next_dest % shovels

def truck_schedule_send_post(truck_id: int, start_area: int, task_id: int) -> int:
    global truck_available_time
    global shovel_available_time
    global last_real_shovel_mass
    global real_shovel_mass
    global dump_available_time
    global last_real_dump_mass
    global real_dump_mass

    print(f'request_truck-{truck_id}-heuristic method-at {env.now}')
    if task_id == 2:
        target = np.argmin((np.maximum(shovel_available_time,
                                       env.now + com_time[start_area][:]) + loading_time - env.now))
    elif task_id == 3:
        target = np.argmin((np.maximum(dump_available_time, env.now + go_time[:, start_area]) + unloading_time - env.now))
    else:
        target = 0
    return target

    # print(f'request_truck-{truck_id}-random method-at {env.now}')
    # if task_id == 2:    # 代表从卸载点到装载点
    #     return np.random.randint(0, shovels, 1)[0]
    #
    # if task_id == 3:    # 代表从装载点到卸载点
    #     return np.random.randint(0, dumps, 1)[0]


def walk_process(env, start_area, truck_id, next_q, direction):
    # 模拟矿卡行走的行为

    global go_to_unload_point_vehical_num
    global go_to_excavator_vehical_num

    global walking_go_to_unload_point_vehical_num
    global walking_go_to_excavator_vehical_num

    global reach_excavator_time_list
    global reach_dump_time_list

    global wating_in_excavator_vehical_num
    global wating_in_unload_point_vehical_num

    global shovel_available_time
    global dump_available_time
    global truck_available_time

    global last_real_shovel_mass
    global last_real_dump_mass

    global real_shovel_mass
    global real_dump_mass

    global goto_excavator_traffic_flow_num

    global goto_dump_traffic_flow_num

    while True:
        task_id = 0
        if "go to unload_point" == direction:
            task_id = 3  # 代表从装载点到卸载点
        elif "go to excavator" == direction:
            task_id = 2  # 代表从卸载点到装载点

        # 进行卡调请求，得到目标电铲/卸载点id
        goal_area = truck_schedule_send_post(truck_id, start_area, task_id)

        global heu_action_time
        heu_action_time = env.now

        # 从数据库中获取行走时长,以及装载/卸载时长
        if "go to excavator" == direction:

            # 此时goal_area代表电铲id，start_area代表卸载点id
            # 本次行走时长，除以60用于将秒换算为分钟

            # walk_time = com_time[start_area][goal_area]

            walk_time = max(1, np.random.normal(com_time[start_area][goal_area], 3))

            # print(f'walk_time: {com_time[start_area][goal_area]} - {walk_time}')

            # 将该条道路的车辆个数加1
            go_to_excavator_vehical_num[start_area][goal_area] = (
                    go_to_excavator_vehical_num[start_area][goal_area] + 1
            )

            # 加入对应挖机抵达列表
            reach_excavator_time_list[goal_area].append(env.now + walk_time)

            # 运输路线车次数加1
            goto_excavator_traffic_flow_num[start_area][goal_area] += 1

            ################################### 关键状态更新 ###################################
            # 更新可用时间
            shovel_available_time[goal_area] = (
                    max(env.now + walk_time, shovel_available_time[goal_area])
                    + loading_time[goal_area]
            )
            truck_available_time[truck_id] = shovel_available_time[goal_area]

            # 产量更新
            last_real_shovel_mass = copy.deepcopy(real_shovel_mass)
            real_shovel_mass[goal_area] += payload

            ####################################################################################

        elif "go to unload_point" == direction:

            # 此时goal_area代表卸载点id，start_area代表电铲id
            # 本次行走时长，除以60用于将秒换算为分钟
            # walk_time = go_time[goal_area][start_area]

            walk_time = max(1, np.random.normal(go_time[goal_area][start_area], 3))

            # print(f'walk_time: {go_time[goal_area][start_area]} - {walk_time}')

            # 将该条道路的车辆个数加1
            go_to_unload_point_vehical_num[start_area][goal_area] = (
                    go_to_unload_point_vehical_num[start_area][goal_area] + 1
            )

            # 加入对应卸点抵达列表
            reach_dump_time_list[goal_area].append(env.now + walk_time)

            # 运输路线车次数加1
            goto_dump_traffic_flow_num[start_area][goal_area] += 1

            ################################### 关键状态更新 ###################################
            # 修改卡车阶段
            # 可用时间更新
            dump_available_time[goal_area] = (
                    max(env.now + walk_time, dump_available_time[goal_area])
                    + unloading_time[goal_area]
            )
            truck_available_time[truck_id] = dump_available_time[goal_area]

            ####################################################################################

        # print(
        #     f"{round(env.now, 2)} - truck_id: {truck_id} - from {start_area} {direction}_{goal_area} - start moving "
        # )

        # 行驶开始，统计在路上行走的车辆个数
        if "go to excavator" == direction:
            # 将该条道路正在行驶的车辆个数加1
            walking_go_to_excavator_vehical_num[start_area][goal_area] = (
                    walking_go_to_excavator_vehical_num[start_area][goal_area] + 1
            )
        elif "go to unload_point" == direction:
            walking_go_to_unload_point_vehical_num[start_area][goal_area] = (
                    walking_go_to_unload_point_vehical_num[start_area][goal_area] + 1
            )

        # 阻塞行走时间
        global request_id
        request_id = yield env.timeout(float(walk_time), value=2)  # 行走时间,单位为分钟

        # for _ in range(int(walk_time)):
        #     yield env.timeout(1)
        #     truck_process += 1

        # 行驶结束，统计在路上行走的车辆个数
        if "go to excavator" == direction:
            # 将该条道路正在行驶的车辆个数减1
            walking_go_to_excavator_vehical_num[start_area][goal_area] = (
                    walking_go_to_excavator_vehical_num[start_area][goal_area] - 1
            )
            # 行走结束，将等待装载的车辆个数加1
            wating_in_excavator_vehical_num[goal_area] = (
                    wating_in_excavator_vehical_num[goal_area] + 1
            )

        elif "go to unload_point" == direction:
            walking_go_to_unload_point_vehical_num[start_area][goal_area] = (
                    walking_go_to_unload_point_vehical_num[start_area][goal_area] - 1
            )
            # 行走结束，将等待卸载的车辆个数加1
            wating_in_unload_point_vehical_num[goal_area] = (
                    wating_in_unload_point_vehical_num[goal_area] + 1
            )

        next_q[goal_area].put(truck_id)  # 将到来的truck放到目标队列中
        # print(
        #     f"{round(env.now, 2)} - truck_id: {truck_id} - {direction}_{goal_area} - end moving"
        # )

        # env.exit()  # 该函数的作用等同于return,直接退出该函数
        return


def excavator_func(env: simpy.Environment, e_q: simpy.Store, u_q, excavator_id):
    # 模拟一个电铲, 一个电铲同时只能处理一台矿卡
    truck_source = simpy.Resource(env, capacity=1)

    global last_real_dump_mass
    global last_real_shovel_mass

    global real_dump_mass
    global real_shovel_mass

    def process(truck_id):
        # 模拟电铲一次工作的进程
        with truck_source.request() as req:
            yield req
            # print(
            #     f"{round(env.now, 2)} - truck_id: {truck_id} - excavator: {excavator_id} - Begin Loading"
            # )

            # 开始装载，记录装载时间
            global load_start_time
            load_start_time[excavator_id] = env.now

            # 开始装载，将等待装载的车辆个数减1
            global wating_in_excavator_vehical_num
            wating_in_excavator_vehical_num[excavator_id] = (
                    wating_in_excavator_vehical_num[excavator_id] - 1
            )

            # 开始装载，将装载车辆个数加1
            global loading_in_excavator_vehical_num
            loading_in_excavator_vehical_num[excavator_id] = (
                    loading_in_excavator_vehical_num[excavator_id] + 1
            )

            # 开始装载，计算装载全部等待车辆时间
            global waiting_vehicle_loaded_time
            waiting_vehicle_loaded_time[excavator_id] = env.now + (wating_in_excavator_vehical_num[excavator_id] + 1) * loading_time[excavator_id]

            # 电铲平均工作装载时间
            # 除以60用于将秒换算为分钟
            # load_time = loading_time[excavator_id]

            load_time = max(1, np.random.normal(loading_time[excavator_id], 1.5))

            # 装载结束，将装载车辆个数减1
            loading_in_excavator_vehical_num[excavator_id] = (
                    loading_in_excavator_vehical_num[excavator_id] - 1
            )

            # 装载结束，将完成装载数加1
            global excavator_loaded_vehicle_num
            excavator_loaded_vehicle_num[excavator_id] = (
                excavator_loaded_vehicle_num[excavator_id] + float(load_time)
            )

            # 装载结束，产量更新
            global real_shovel_mass
            global last_real_shovel_mass
            last_real_shovel_mass = copy.deepcopy(real_shovel_mass)
            real_shovel_mass[excavator_id] += payload

            print(
                f"{round(env.now, 2)} - truck_id: {truck_id} - excavator: {excavator_id} - End Loading"
                " - Request Dispatching"
            )

            # 矿卡从电铲处行走到卸载点
            env.process(
                walk_process(env, excavator_id, truck_id, u_q, "go to unload_point")
            )

    while True:
        truck_id = yield e_q.get()
        env.process(process(truck_id))


def unloadpoint_func(env: simpy.Environment, u_q: simpy.Store, e_q, unload_point_id):
    # 模拟一个卸载点, 一个卸载点同时只能处理一台矿卡
    truck_source = simpy.Resource(env, capacity=1)

    global real_dump_mass
    global real_shovel_mass
    global truck_location
    global cycle_time

    def process(truck_id):
        # 模拟卸载点一次工作的进程"""
        with truck_source.request() as req:
            yield req
            print(
                f"{round(env.now, 2)} - truck_id: {truck_id} - UnloadPoint: {unload_point_id} - Begin Unloading"
            )

            # 开始卸载，将等待卸载的车辆个数减1
            global wating_in_unload_point_vehical_num
            wating_in_unload_point_vehical_num[unload_point_id] = (
                    wating_in_unload_point_vehical_num[unload_point_id] - 1
            )

            # 开始卸载，将正在的卸载车辆个数加1
            global unloading_in_unload_point_vehical_num
            unloading_in_unload_point_vehical_num[unload_point_id] = (
                    unloading_in_unload_point_vehical_num[unload_point_id] + 1
            )

            # 开始卸载，计算装载全部车辆时间
            global waiting_vehicle_unloaded_time
            waiting_vehicle_unloaded_time[unload_point_id] = \
                env.now + (wating_in_unload_point_vehical_num[unload_point_id] + 1) * unloading_time[unload_point_id]


            # 卸载点平均工作卸载时间
            # 除以60用于将秒换算为分钟
            # unload_time = unloading_time[unload_point_id]

            unload_time = max(1, np.random.normal(unloading_time[unload_point_id], 1.5))

            # 阻塞卸载时间
            global request_id
            # yield env.timeout(float(unload_time))  # 进行卸载操作
            request_id = yield env.timeout(float(unload_time), value=1)  # 进行卸载操作

            # 卸载结束，将卸载车辆个数减1
            unloading_in_unload_point_vehical_num[unload_point_id] = (
                    unloading_in_unload_point_vehical_num[unload_point_id] - 1
            )

            # 卸载结束，将完成卸载数加1
            global dump_loaded_vehicle_num
            dump_loaded_vehicle_num[unload_point_id] = (
                excavator_loaded_vehicle_num[unload_point_id] + float(unload_time)
            )

            # 卸载结束，产量更新
            global real_dump_mass
            global last_real_dump_mass
            last_real_dump_mass = copy.deepcopy(real_dump_mass)
            real_dump_mass[unload_point_id] += payload

            print(
                f"{round(env.now, 2)} - truck_id: {truck_id} - UnloadPoint: {unload_point_id} - End Unloading"
                " - Request Dispatching"
            )

            # 矿卡从卸载点处行走到电铲
            env.process(
                walk_process(env, unload_point_id, truck_id, e_q, "go to excavator")
            )

    while True:
        truck_id = yield u_q.get()
        env.process(process(truck_id))


# 在停车场按照固定时间生成一定数量的矿卡
def generate_truck_in_parking_lot(env, e_q, u_q):
    global shovel_available_time
    global dump_available_time
    global truck_available_time
    global last_real_shovel_mass
    global real_shovel_mass
    global cycle_time_start

    def process(truck_id, walk_time, goal_area, e_q):
        ################################### 关键状态更新 ###################################
        # 更新电铲，矿卡可用时间
        shovel_available_time[goal_area] = (
                max(env.now + walk_time, shovel_available_time[goal_area])
                + loading_time[goal_area]
        )
        truck_available_time[truck_id] = shovel_available_time[goal_area]

        go_to_excavator_vehical_num[0][goal_area] = (
                go_to_excavator_vehical_num[0][goal_area] + 1
        )

        # 阻塞行走时间
        global request_id
        request_id = yield env.timeout(float(walk_time), value=2)  # 行走时间,单位为分钟

        # 产量更新
        last_real_shovel_mass = copy.deepcopy(real_shovel_mass)
        real_shovel_mass[goal_area] += payload

        e_q[goal_area].put(truck_id)  # 将到来的truck放到电铲的队列中

        # 行走结束，将等待装载的车辆个数加1
        wating_in_excavator_vehical_num[goal_area] = (
                wating_in_excavator_vehical_num[goal_area] + 1
        )

        print(
            f"{round(env.now, 2)} - truck_id: {truck_id} - From Parking Lot to WorkArea:{goal_area} end moving"
        )

    for i in range(truck_num):
        # 模拟矿卡随机请求调度
        t = 1  # 固定停1*60=60秒

        # 阻塞行走时间
        global request_id
        request_id = yield env.timeout(t, value=2)

        global truck_process

        task_id = 1  # task_id等于1，说明是从停车场到装载点

        target = truck_schedule_send_post_start(i, 0, task_id, env)  # 得到电铲id

        # 本次行走时长
        walk_time = com_time[0][target]

        print(
            f"{round(env.now, 2)} - truck_id: {i} - From Parking Lot to WorkArea:{target} start moving "
        )

        env.process(process(i, walk_time, target, e_q))

env = simpy.Environment()

def env_reset(rl_mode=False):
    global go_to_unload_point_vehical_num  # 用于保存每条卸载道路上经过的车辆个数
    global go_to_excavator_vehical_num  # 用于保存每条装载道路上的经过的车辆总个数
    global sim_start_time  # 用于保存仿真程序运行的真实起始时间

    global walking_go_to_unload_point_vehical_num  # 卸载道路上正在行走的车辆个数
    global walking_go_to_excavator_vehical_num  # 装载道路上正在行走的车辆个数

    global reach_excavator_time_list  # 驶往挖机矿卡预期抵达时间
    global reach_dump_time_list  # 驶往卸点预期抵达时间

    global loading_in_excavator_vehical_num  # 挖机装车数
    global unloading_in_unload_point_vehical_num  # 卸点卸车数

    global wating_in_excavator_vehical_num  # 挖机排队等待装载矿卡数
    global wating_in_unload_point_vehical_num  # 卸点排队等待卸载矿卡数

    global excavator_loaded_vehicle_num  # 挖机完成装载的车次数
    global dump_loaded_vehicle_num  # 卸点完成卸载的车次数

    global waiting_vehicle_loaded_time  # 完成所有排队等待车辆装载时间
    global waiting_vehicle_unloaded_time  # 完成所有排队等待车辆卸载时间

    global dump_available_time  # 卸点最近一次可用时间
    global shovel_available_time  # 挖机最近一次可用时间
    global truck_available_time  # 矿卡最近一次可用时间

    global real_shovel_mass  # 挖机实际装载量
    global real_dump_mass  # 卸点实际卸载量

    global last_real_shovel_mass  # 上一次调度任务挖机实际装载量
    global last_real_dump_mass  # 上一次调度任务卸点实际卸载量

    global recording  # 数据统计{时间(min), 产量(tonnes), 油耗(liters)}

    global goto_excavator_traffic_flow_num  # 驶往挖机各运输路线实际车次
    global goto_dump_traffic_flow_num  # 驶往卸点各运输路线实际车次

    global load_start_time  # 挖机开始装载时间
    global unload_start_time  # 卸点开始卸载时间

    global load_end_time  # 挖机结束装载时间
    global unload_end_time  # 卸点结束装载时间

    global cycle_time_start  # 统计各车辆行程开始时间

    global cycle_time_list  # 统计行程时长

    global rl_truck

    global rl_allow

    global rl_evaluate

    global heu_rpm

    global heu_action

    global heu_action_time

    # 实例环境
    global env
    env = simpy.Environment()

    # 获取装载点和卸载点的个数
    num_of_load_area = shovels
    num_of_unload_area = dumps

    e_q = []
    for _ in range(num_of_load_area):
        e_q.append(simpy.Store(env))

    u_q = []
    for _ in range(num_of_unload_area):
        u_q.append(simpy.Store(env))

    # 保存每条道路的车辆个数
    go_to_unload_point_vehical_num = np.zeros((num_of_load_area, num_of_unload_area))
    go_to_excavator_vehical_num = np.zeros((num_of_unload_area, num_of_load_area))

    # real_comp_workload = np.zeros(dumps)

    # 统计在路上行驶的车辆个数
    walking_go_to_unload_point_vehical_num = np.zeros(
        (num_of_load_area, num_of_unload_area)
    )
    walking_go_to_excavator_vehical_num = np.zeros(
        (num_of_unload_area, num_of_load_area)
    )

    # 初始化车辆抵达列表
    reach_excavator_time_list = [[] for _ in range(shovels)]   # 驶往挖机矿卡预期抵达时间
    reach_dump_time_list = [[] for _ in range(dumps)]  # 驶往卸点预期抵达时间

    # 统计正在装载或者卸载的车辆个数
    loading_in_excavator_vehical_num = np.zeros(num_of_load_area)
    unloading_in_unload_point_vehical_num = np.zeros(num_of_unload_area)

    # 统计正在排队的车辆个数
    wating_in_excavator_vehical_num = np.zeros(num_of_load_area)
    wating_in_unload_point_vehical_num = np.zeros(num_of_unload_area)

    # 初始化装卸载完成车次数
    excavator_loaded_vehicle_num = np.zeros(num_of_load_area)
    dump_loaded_vehicle_num = np.zeros(num_of_unload_area)

    # 初始化完成排队车辆服务时间
    waiting_vehicle_loaded_time = np.zeros(shovels)
    waiting_vehicle_unloaded_time = np.zeros(dumps)

    # 初始化设备可用时间
    dump_available_time = np.zeros(num_of_unload_area)
    shovel_available_time = np.zeros(num_of_load_area)
    truck_available_time = np.zeros(truck_num)

    # 初始化实时产量
    last_real_shovel_mass = np.zeros(num_of_load_area)
    last_real_dump_mass = np.zeros(num_of_unload_area)

    real_shovel_mass = np.zeros(num_of_load_area)
    real_dump_mass = np.zeros(num_of_unload_area)

    # 初始化统计表
    recording = [[0, 0, 0]]

    # 初始化各路线车次数
    goto_excavator_traffic_flow_num = np.zeros((dumps, shovels))
    goto_dump_traffic_flow_num = np.zeros((shovels, dumps))

    load_start_time = np.zeros((shovels, ), dtype=float)
    unload_start_time = np.zeros((dumps, ), dtype=float)

    load_end_time = np.zeros((shovels, ), dtype=float)
    unload_end_time = np.zeros((dumps, ), dtype=float)

    cycle_time_start = np.zeros((truck_num, ), dtype=float)

    cycle_time_list = []

    # 启动挖机及卸点进程
    for i in range(num_of_load_area):
        env.process(excavator_func(env, e_q[i], u_q, excavator_id=i))

    for i in range(num_of_unload_area):
        env.process(unloadpoint_func(env, u_q[i], e_q, unload_point_id=i))

    # 从停车位开始向电铲派车
    env.process(generate_truck_in_parking_lot(env, e_q, u_q))


env_reset()

env.run(480)

print(real_dump_mass)

