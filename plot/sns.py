#!E:\Pycharm Projects\Waytous
# -*- coding: utf-8 -*-
# @Time : 2022/11/11 14:50
# @Author : Opfer
# @Site :
# @File : sns.py    
# @Software: PyCharm

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

file_path1 = r'../results/hyb'
file_path2 = r'../results/sim'

files = [file_path1, file_path2]

features = ["train_reward"]


def smooth(data, sm=1):
    if sm > 1:
        smooth_data = []
        for d in data:
            y = np.ones(sm) * 1.0 / sm
            d = np.convolve(y, d, "same")

            smooth_data.append(d)

        return smooth_data
    return data


def get_file_path(dict_path=None):

    csv_files = []

    for root, dirs, files in os.walk(dict_path):
        for file in files:
            path = os.path.join(root, file)
            csv_files.append(path)

    return csv_files


def get_data(file_path=None):

    data_files = get_file_path(file_path)

    data_set = []

    for csv_file in data_files:
        data = pd.read_csv(csv_file)
        data = pd.DataFrame(data, columns=features)
        data_set.append(data.values.transpose())

    data = np.vstack(data_set)

    data = smooth(data, 100)

    return data

# simulation results

lables = ["hyd", "sim"]

df = []

for i in range(2):

    data = get_data(files[i])

    data_df = pd.DataFrame(data).melt(var_name='episode', value_name='reward')

    data_df['episode'] = data_df['episode'] * 5

    df.append(data_df)

    df[i]['algo']= lables[i]

df = pd.concat(df)

# df['episode'] = df['episode'] * 5

# df = df[:10000]

g = sns.lineplot(x="episode", y="reward", hue="algo", style="algo", data=df)

xlabels = ['{:,.1f}'.format(x) + 'K' for x in g.get_xticks()/1000]

g.set_xticklabels(xlabels)

plt.show()