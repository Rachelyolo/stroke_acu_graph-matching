import numpy as np

import os
from tqdm import tqdm

from scipy.stats import pearsonr
from numpy.linalg import inv
import matplotlib.pyplot as plt

# for i in range(health_num):
#     plt.figure()
#     plt.scatter(samplex[i], sampley[i], 5)
#     ax = plt.gca()  # 获取到当前坐标轴信息
#     ax.xaxis.set_ticks_position('top')  # 将X坐标轴移到上面
#     ax.invert_yaxis()
#     file = "../fmri_permutation_output/stroke_Person_Z_" + str(i + 1) + ".png"
#     plt.savefig(file)
# print("------output heatmap--------")
import pandas as pd

father_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "443")
father_path = os.path.join(father_path, "connectome_fmri_o_443")
t0 = "stroke-flip-post"  # lack 19
t1 = "stroke-flip-pre"  # lack 19 28

# N = 0.1  # therold
dim = 443
head_num = 18  # 脑区个数
health_num = 26 - 1


def precision(matrix, health_num, dim, gamma):
    correlation_matrix = np.zeros([health_num, dim, dim])
    precision_matrix = np.zeros([health_num, dim, dim])
    reg = np.zeros([matrix.shape[0], dim, dim])
    # all_mse = []
    # opt_gamma = 0
    # min_rmse = float("inf")
    # gamma_s = np.linspace(0, 1, 50)
    # idx_min = 0
    # gamma_t = [gamma]
    # for idx, gamma in enumerate(gamma_t):
    #     pre = []
    #     inverse = []
    #     diff = []
    #     for i in tqdm(range(matrix.shape[0])):
    #         np.fill_diagonal(matrix[i], 0)
    #         a1 = matrix[i] + gamma * np.eye(dim)
    #         pre1 = inv(a1)
    #         pre2 = inv(matrix[i])
    #         pre.append(pre1)
    #         inverse.append(pre2)
    #     group_prec = np.mean(inverse, axis=0)
    #     for i in range(matrix.shape[0]):
    #         diff.append(
    #             np.linalg.norm(pre[i][np.triu_indices(dim, 1)] - group_prec[np.triu_indices(dim, 1)]))
    #     rmse = np.mean(diff)
    #     print("rmse", rmse)
    #     all_mse.append(rmse)
    #     if rmse<min_rmse:
    #         opt_gamma = gamma
    #         min_rmse = rmse
    #         idx_min = idx
    # print("opt", opt_gamma)
    # for i in range(matrix.shape[0]):
    for i in tqdm(range(matrix.shape[0])):
        np.fill_diagonal(matrix[i], 0)
        correlation_matrix = np.zeros([matrix[i].shape[1], matrix[i].shape[1]])
        for ii in range(matrix.shape[1]):
            for j in range(matrix.shape[1]):
                correlation_matrix[ii, j], _ = pearsonr(matrix[i][:, ii], matrix[i][:, j])
        a1 = correlation_matrix + gamma * np.eye(dim)
        pre1 = inv(a1)
        precision_matrix[i] = pre1

    return precision_matrix


sett = [8, 8, 10, 10, 7, 30, 40, 23, 25, 13, 30, 39, 30, 40, 23, 25, 13, 30, 39]
# 1和3 放到最后
# 2和4 放到之前的11后面


# 1-8；9-16；17-26；27-36；44-73；74-113；114-136；137-161；162-174；175-204；205-243
network_trans = np.zeros([head_num, head_num])

time_0_path_health = os.path.join(father_path, t0)
time_1_path_health = os.path.join(father_path, t1)

from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from matplotlib import pyplot as plt

data_all = []
flip_or_not = []
for path, _, files in os.walk(time_0_path_health):
    # print(path, files)
    for idx, file in enumerate(files):
        if file.endswith('.mat') or "19" in file or "28" in file or "02" in file:
            # if file.endswith('.mat') or "19" in file:
            continue
        file_name = os.path.join(path, file)
        # print(file_name)
        with open(file_name, "r") as f:
            data = []
            for iss, line in enumerate(f.readlines()):
                s = line.split(' ')
                ss = []
                for s_ in s:
                    if s_:
                        ss.append(float(s_))
                # print(len(ss))
                ss = np.array(ss)
                data.append(ss)
            data = np.stack(data, axis=1)
        data_all.append(data)
data_all = np.stack(data_all, axis=0)
data_all_2 = []
for path, _, files in os.walk(time_1_path_health):
    for idx, file in enumerate(files):
        if file.endswith('.mat') or "19" in file or "28" in file or "02" in file:
            # if file.endswith('.mat') or "19" in file:
            continue
        file_name = os.path.join(path, file)
        with open(file_name, "r") as f:
            data = []
            for iss, line in enumerate(f.readlines()):
                s = line.split(' ')
                ss = []
                for s_ in s:
                    if s_:
                        ss.append(float(s_))
                # print(len(ss))
                ss = np.array(ss)
                data.append(ss)
            data = np.stack(data, axis=1)
        data_all_2.append(data)

data_all_2 = np.stack(data_all_2, axis=0)
cost_map_all = np.zeros([health_num, dim, dim])
data_all_2[np.isnan(data_all_2)] = 0
data_all_2[np.isinf(data_all_2)] = 0
data_all[np.isnan(data_all)] = 0
data_all[np.isinf(data_all)] = 0


# np.save("../fmri_precision_output/stroke_post_before_precision_Z.npy", data_all)
# np.save("../fmri_precision_output/stroke_pre_before_precision_Z.npy", data_all_2)

#
# file = open("../fmri_precision_output/stroke_post_before_precision_Z.txt", "w")
# for sub in range(data_all.shape[0]):
#     num = str(sub+1) + '\n'
#     file.writelines(num)
#     for i in range(data_all[sub].shape[0]):
#         file.writelines(str(data_all[sub][i]) + '\n')
#
# file = open("../fmri_precision_output/stroke_pre_before_precision_Z.txt", "w")
# for sub in range(data_all_2.shape[0]):
#     num = str(sub+1) + '\n'
#     file.writelines(num)
#     for i in range(data_all_2[sub].shape[0]):
#         file.writelines(str(data_all_2[sub][i]) + '\n')

def calculate_cost(fc1, fc2):
    # number of brain regions
    nROIs = fc1.shape[0]
    # initialize empty matrix
    costmat = np.zeros((nROIs, nROIs))
    for x in range(0, nROIs):  # x = time point 1.
        a = fc1[x]

        for y in range(0, nROIs):  # y = time point 2.
            b = fc2[y]
            # cost to assign node x in fc1 to node y in fc2.
            costmat[x, y] = distance.euclidean(a, b)
    return costmat


data_all_o = data_all
data_all_o_2 = data_all_2
for idx, gamma in enumerate(np.linspace(0.0, 1.0, 50)):
    print(idx)
    data_all = precision(data_all_o, health_num, dim, gamma)
    data_all_2 = precision(data_all_o_2, health_num, dim, gamma)
    # np.save("../fmri_precision_output/stroke_pre_before_precision.npy", data_all)
    # np.save("../fmri_precision_output/stroke_pre_after_precision.npy", data_all_2)
    #
    #
    # data_all = np.load("../fmri_precision_output/stroke_post_before_precision_Z.npy")
    # data_all_2 = np.load("../fmri_precision_output/stroke_pre_before_precision_Z.npy")

    health_num = data_all.shape[0]
    dim = data_all.shape[1]
    cost_map_all = np.zeros([health_num, dim, dim])
    for subject_x in range(data_all.shape[0]):
        costmap = calculate_cost(data_all[subject_x], data_all_2[subject_x])
        cost_map_all[subject_x] = costmap

    # file = open("../fmri_costmap_output/before_costmap_stroke_Z.txt", "w")
    # for sub in range(cost_map_all.shape[0]):
    #     num = str(sub + 1) + '\n'
    #     file.writelines(num)
    #     for i in range(cost_map_all[sub].shape[0]):
    #         file.writelines(str(cost_map_all[sub][i]) + '\n')
    # np.save("../fmri_costmap_output/before_costmap_stroke_Z.npy", cost_map_all)
    print("finish save costmap")

    dict_sample_s = {"data": [], "type": [], "x": []}
    cost_map_all_s = cost_map_all
    dim = cost_map_all_s.shape[1]

    data_frame_s = pd.DataFrame(dict_sample_s)
    health_num = cost_map_all_s.shape[0]
    transform_matrix_s = np.zeros([cost_map_all_s.shape[0], cost_map_all_s.shape[1], cost_map_all_s.shape[2]])
    head_num = 18
    network_trans = np.zeros([head_num, head_num])
    network_trans_num = np.zeros([head_num, head_num])
    samplex = []
    sampley = []

    for sub in range(cost_map_all_s.shape[0]):  # 第i个病人
        x_sample = []
        y_sample = []
        cost = cost_map_all_s[sub]
        rowind, colind = linear_sum_assignment(cost)
        for index in rowind:
            x_sample.append(index)
            y_sample.append(colind[index])
            transform_matrix_s[sub][index, colind[index]] = 1
        sub_list = [str(sub)] * dim
        dict_sample_s['data'].extend(y_sample)
        dict_sample_s['x'].extend(x_sample)
        dict_sample_s['type'].extend(sub_list)
        samplex.append(x_sample)
        sampley.append(y_sample)
    data_frame_s = pd.DataFrame(dict_sample_s)
    sample_y_all = []
    temp = 0
    sett = [8, 8, 10, 10, 7, 30, 40, 23, 25, 13, 30, 39, 30, 40, 23, 25, 13, 30, 39]
    transform_matrix_all = np.sum(transform_matrix_s, axis=0)
    # 1和3 放到最后
    # 2和4 放到之前的11后
    network_load = {}
    label = 1
    start = 0
    for i in sett:
        if i == 7:
            start += i
            continue
        for idx, j in enumerate(range(start, i + start)):
            if idx == 0:
                network_load[label] = [start]
            else:
                network_load[label].append(j)
        start += i
        label += 1

    temp_ls = 0
    mm = 0

    x_tick = ['5', '6', '7', '8', '9', '10', '11', '2', '4', '12', '13', '14', '15', '16', '17', '18', '1', '3']
    y_tick = ['5', '6', '7', '8', '9', '10', '11', '2', '4', '12', '13', '14', '15', '16', '17', '18', '1', '3']

    for idx, set_order in enumerate(x_tick):
        current_network = idx
        network_dim = network_load[int(set_order)]
        for small_dim_curr in network_dim:
            for idx2, set_order2 in enumerate(x_tick):
                change_network = idx2
                network_dum = network_load[int(set_order2)]
                for small_dim_change in network_dum:
                    if small_dim_change == small_dim_curr:
                        continue
                    network_trans[current_network, change_network] += transform_matrix_all[small_dim_curr][
                                                                          small_dim_change] / (
                                                                              len(network_dim) *
                                                                              transform_matrix_s.shape[0])
                    network_trans_num[current_network, change_network] += transform_matrix_all[small_dim_curr][
                        small_dim_change]
    # print(network_trans)

    # np.save("../fmri_heatmap_output/fmri_before_stroke_network_Z.npy", network_trans)

    x_tick = ['R visual', 'R somatosensory', 'R dorsal attention', 'R ventral attention', 'R limbic',
              'R fronto-parietal',
              'R default mode', 'R subcortical', 'R cerebellum', 'L visual', 'L somatosensory', 'L dorsal attention',
              'L ventral attention', 'L limbic', 'L fronto-parietal', 'L default mode', 'L subcortical', 'L cerebellum']
    y_tick = ['R visual', 'R somatosensory', 'R dorsal attention', 'R ventral attention', 'R limbic',
              'R fronto-parietal',
              'R default mode', 'R subcortical', 'R cerebellum', 'L visual', 'L somatosensory', 'L dorsal attention',
              'L ventral attention', 'L limbic', 'L fronto-parietal', 'L default mode', 'L subcortical', 'L cerebellum']
    for idx, i in enumerate(x_tick):
        x_tick[idx] = i.replace(' ', '-')
    for idx, j in enumerate(y_tick):
        y_tick[idx] = j.replace(' ', '-')

    data = {}

    for i in range(18):
        data[x_tick[i]] = network_trans[i]
    plt.figure(figsize=(12, 11))
    import seaborn as sns

    pd_data = pd.DataFrame(data, index=y_tick, columns=x_tick)
    sns.heatmap(data=pd_data, square=True)
    plt.savefig(
        "../fmri_heatmap_output_test_stroke2/fmri_after_stroke_correlation_heatmap_percent_" + str(gamma) + ".png")

    data = {}

    # for i in range(18):
    #     data[x_tick[i]] = network_trans_num[i]
    # plt.figure(figsize=(12, 11))
    #
    # pd_data = pd.DataFrame(data, index=y_tick, columns=x_tick)
    # sns.heatmap(data=pd_data, square=True)
    # plt.savefig("../fmri_heatmap_output_test_stroke/fmri_before_stroke_heatmap_num_"+str(gamma)+".png")

print("finish job, save origin data")