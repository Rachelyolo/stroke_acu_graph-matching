import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from matplotlib import pyplot as plt

dict_sample_s = {"data":[], "type":[], "x":[]}
cost_map_all_s = np.load("../dti_costmap_output/dti_flip_health_costmap.npy")
dim = cost_map_all_s.shape[1]
data_frame_s = pd.DataFrame(dict_sample_s)
health_num = cost_map_all_s.shape[0]
transform_matrix_s = np.zeros([cost_map_all_s.shape[0], cost_map_all_s.shape[1], cost_map_all_s.shape[2]])
head_num = 18
network_trans = np.zeros([head_num, head_num])
network_trans_num = np.zeros([head_num, head_num])
samplex = []
sampley = []
num_count = 0
people = np.zeros([cost_map_all_s.shape[0], 1])
for sub in range(cost_map_all_s.shape[0]): # 第i个病人
    num_count = 0
    x_sample = []
    y_sample = []
    cost = cost_map_all_s[sub]

    rowind, colind = linear_sum_assignment(cost)
    for index in rowind:
        x_sample.append(index)
        y_sample.append(colind[index])
        if index != colind[index]:
            num_count += 1
        transform_matrix_s[sub][index, colind[index]] = 1
    sub_list = [str(sub)] * dim
    dict_sample_s['data'].extend(y_sample)
    dict_sample_s['x'].extend(x_sample)
    dict_sample_s['type'].extend(sub_list)
    people[sub] = num_count
    samplex.append(x_sample)
    sampley.append(y_sample)
print("------output heatmap--------")
data_frame_s = pd.DataFrame(dict_sample_s)
sample_y_all = []
temp = 0
sett = [8, 8, 10, 10, 7, 30, 40, 23, 25, 13, 30, 39, 30, 40, 23, 25, 13, 30, 39]
transform_matrix_all = np.sum(transform_matrix_s, axis=0)
people = pd.DataFrame(people)
people.to_excel("../num/dti_health_num.xlsx")
network_load = {}
label = 1
start = 0
for i in sett:
    if i == 7:
        start += i
        continue
    for idx, j in enumerate(range(start, i+start)):
        if idx == 0:
            network_load[label] = [start]
        else:
            network_load[label].append(j)
    start += i
    label+=1



for i in range(transform_matrix_s.shape[0]):
    network_trans = np.zeros([head_num, head_num])
    network_trans_num = np.zeros([head_num, head_num])
    temp_ls = 0
    mm = 0

    x_tick = ['5', '6', '7', '8', '9', '10', '11', '2', '4', '12', '13', '14', '15', '16', '17', '18', '1', '3']
    y_tick = ['5', '6', '7', '8', '9', '10', '11', '2', '4', '12', '13', '14', '15', '16', '17', '18', '1', '3']
    transform_matrix_all = transform_matrix_s[i]
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
    network_trans_save = pd.DataFrame(network_trans)
    path = "../csv_output_dti_health/"
    if i<10:
        network_trans_save.to_csv(path + "0"+str(i) +"dti_health.csv")
    else:
        network_trans_save.to_csv(path + str(i) + "dti_health.csv")