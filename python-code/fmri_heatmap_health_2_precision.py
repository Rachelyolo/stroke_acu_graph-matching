import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from matplotlib import pyplot as plt


def run(gamma, path):
    dict_sample_s = {"data": [], "type": [], "x": []}
    # path = "../fmri_costmap_output/before_costmap_health_" + str(gamma) + ".npy"
    cost_map_all_s = np.load(path)
    dim = cost_map_all_s.shape[1]

    data_frame_s = pd.DataFrame(dict_sample_s)
    health_num = cost_map_all_s.shape[0]
    transform_matrix_s = np.zeros([cost_map_all_s.shape[0], cost_map_all_s.shape[1], cost_map_all_s.shape[2]])
    head_num = 18 - 2
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

    import matplotlib.pyplot as plt
    # for i in range(health_num):
    #     plt.figure()
    #     plt.scatter(samplex[i], sampley[i], 5)
    #     ax = plt.gca()                                 #获取到当前坐标轴信息
    #     ax.xaxis.set_ticks_position('top')   #将X坐标轴移到上面
    #     ax.invert_yaxis()
    #     file = "../fmri_permutation_output/health_Person_Z_precision" + str(i+1) + ".png"
    #     plt.savefig(file)
    # print("------output heatmap--------")
    # data_frame_s = pd.DataFrame(dict_sample_s)
    # sample_y_all = []
    # temp = 0
    sett = [7, 30, 40, 23, 25, 13, 30, 39, 8, 10, 30, 40, 23, 25, 13, 30, 39, 8, 10]
    transform_matrix_all = np.sum(transform_matrix_s, axis=0)


    path1 = "../threold_research/fmri_health_node_" + str(gamma) + ".npy"
    np.save(path1, transform_matrix_all)
    print("所有健康人fmri变换综合的443*443矩阵保存在：", path1)
    print("save 443*443 node successfully")
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

    # network_load_new = {}
    # for key in network_load.keys():
    #     network_load_new[key] = network_load[key]
    # network_load_new[8] = network_load[8]
    # network_load_new[8].extend(network_load[9])
    # network_load_new[17] = network_load[17]
    # network_load_new[17].extend(network_load[18])
    network_load_new = {}
    for key in network_load.keys():
        if key <= 4:
            continue
        network_load_new[key] = network_load[key]
    network_load_new[1] = network_load[1]
    network_load_new[1].extend(network_load[3])
    network_load_new[2] = network_load[2]
    network_load_new[2].extend(network_load[4])
    # x_tick = ['1', '2', '3', '4', '5', '6', '7', '8', '10', '11', '12', '13', '14', '15', '16', '17']


    x_tick = ['5', '6', '7', '8', '9', '10', '11', '2', '12', '13', '14', '15', '16', '17', '18', '1']

    for idx, set_order in enumerate(x_tick):
        current_network = idx
        network_dim = network_load_new[int(set_order)]
        for small_dim_curr in network_dim:
            for idx2, set_order2 in enumerate(x_tick):
                change_network = idx2
                network_dum = network_load_new[int(set_order2)]
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

    path2 = "../threold_research/fmri_health_network_" + str(gamma) + ".npy"

    np.save(path2, network_trans)
    print("所有健康人变换综合的16*16矩阵(脑网络frequency文件)保存在：", path2)
    return path1, path2

    #
    # # np.save("../fmri_heatmap_output/fmri_after_health_network_precision.npy", network_trans)
    #
    # x_tick = ['R visual', 'R somatosensory', 'R dorsal attention', 'R ventral attention', 'R limbic',
    #           'R fronto-parietal',
    #           'R default mode', 'R subcortical', 'R cerebellum', 'L visual', 'L somatosensory', 'L dorsal attention',
    #           'L ventral attention', 'L limbic', 'L fronto-parietal', 'L default mode', 'L subcortical', 'L cerebellum']
    # y_tick = ['R visual', 'R somatosensory', 'R dorsal attention', 'R ventral attention', 'R limbic',
    #           'R fronto-parietal',
    #           'R default mode', 'R subcortical', 'R cerebellum', 'L visual', 'L somatosensory', 'L dorsal attention',
    #           'L ventral attention', 'L limbic', 'L fronto-parietal', 'L default mode', 'L subcortical', 'L cerebellum']
    # for idx, i in enumerate(x_tick):
    #     x_tick[idx] = i.replace(' ', '-')
    # for idx, j in enumerate(y_tick):
    #     y_tick[idx] = j.replace(' ', '-')
    #
    #
    # data = {}

    # for i in range(18):
    #     data[x_tick[i]] = network_trans[i]
    # plt.figure(figsize=(12, 11))
    # import seaborn as sns
    #
    # pd_data = pd.DataFrame(data, index=y_tick, columns=x_tick)
    # sns.heatmap(data=pd_data, square=True)
    # plt.savefig("../fmri_heatmap_output/fmri_after_health_heatmap_percent_precision.png")
    #
    # data = {}
    #
    # for i in range(18):
    #     data[x_tick[i]] = network_trans_num[i]
    # plt.figure(figsize=(12, 11))
    #
    # pd_data = pd.DataFrame(data, index=y_tick, columns=x_tick)
    # sns.heatmap(data=pd_data, square=True)
    #
    # plt.savefig("../fmri_heatmap_output/fmri_after_health_heatmap_num_precision.png")