
import numpy as np

import os
from tqdm import tqdm

from numpy.linalg import inv
def run(gamma):
    father_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "443")
    father_path = os.path.join(father_path, "connectome_fmri_443")
    t0 = "health-post"# lack 1 17
    t1 = "health-pre" # lack



    name = "_Z"
    # N = 0.1  # therold
    dim = 443
    head_num = 18 # 脑区个数
    health_num = 22

    def precision(matrix, health_num, dim,gamma):
        correlation_matrix = np.zeros([health_num, dim, dim])
        precision_matrix = np.zeros([health_num, dim, dim])

        # reg = np.zeros([matrix.shape[0], dim, dim])
        # all_mse = []
        # opt_gamma = 0
        # min_rmse = float("inf")
        # gamma_s = np.linspace(0, 1, 50)
        # idx_min = 0
        # for idx, gamma in enumerate(np.linspace(0, 1, 50)):
        #     print(gamma)
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
        # # for i in range(matrix.shape[0]):
        for i in tqdm(range(matrix.shape[0])):
            # np.fill_diagonal(matrix[i], 0)
            cov = np.cov(matrix[i])
            a1 = matrix[i]
            pre1 = inv(cov)
            # precision_matrix[i] = pre1
            precision_matrix[i] = pre1

        return precision_matrix

    sett = [8, 8, 10, 10, 7, 30, 40, 23, 25, 13, 30, 39, 30, 40, 23, 25, 13, 30, 39]
    # 1和3 放到最后
    # 2和4 放到之前的11后面



    # 1-8；9-16；17-26；27-36；44-73；74-113；114-136；137-161；162-174；175-204；205-243
    network_trans = np.zeros([head_num, head_num])

    time_0_path_health = os.path.join(father_path, t0)
    time_1_path_health = os.path.join(father_path, t1)
    print("读取文件夹:", time_0_path_health, " 以及 ", time_1_path_health)





    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import distance
    from matplotlib import pyplot as plt
    data_all = []
    flip_or_not = []
    for path, _, files in os.walk(time_0_path_health):
        files.sort(key=lambda x: int(x[26:28]))
        # print(files)

        for idx, file in enumerate(files):
            if file.endswith('.mat') or "01" in file or "17" in file:
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
            data_all.append(data)
    data_all = np.stack(data_all, axis=0)
    data_all_2 = []
    for path, _, files in os.walk(time_1_path_health):
        files.sort(key=lambda x: int(x[26:28]))
        for idx, file in enumerate(files):
            if file.endswith('.mat') or "01" in file or "17" in file:
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
    print("总共读取的人数: ", data_all.shape[0])



    # no precision part

    path1 = "../fmri_precision_output/health_post_before_precision_" + str(gamma) + ".npy"
    print("health post文件保存在(无precision)：", path1)
    print(data_all.shape)
    sett = [8, 8, 10, 10, 7, 30, 40, 23, 25, 13, 30, 39, 30, 40, 23, 25, 13, 30, 39]
    # sett = [7, 30, 40, 23, 25, 13, 30, 39, 30, 40, 23, 25, 13, 30, 39]
    # data_all_new = np.zeros_like(data_all)
    data_all_new = np.zeros_like(data_all)
    for sub in range(data_all.shape[0]):
        data_all_new[sub][425: 433] = data_all[sub][0: 8]
        data_all_new[sub][433: ] = data_all[sub][16: 26]
        data_all_new[sub][207:215] = data_all[sub][8:16]
        data_all_new[sub][215:225] = data_all[sub][26:36]
        data_all_new[sub][0:207] = data_all[sub][36: 243]
        data_all_new[sub][225: 425] = data_all[sub][243: ]
        data_all_new[sub][:, 425: 433] = data_all[sub][:, 0: 8]
        data_all_new[sub][:, 433:] = data_all[sub][:, 16: 26]
        data_all_new[sub][:, 207:215] = data_all[sub][:, 8:16]
        data_all_new[sub][:, 215:225] = data_all[sub][:, 26:36]
        data_all_new[sub][:, 0:207] = data_all[sub][:, 36: 243]
        data_all_new[sub][:, 225: 425] = data_all[sub][:, 243:]
    data_all_new_2 = np.zeros_like(data_all)
    for sub in range(data_all.shape[0]):
        data_all_new_2[sub][425: 433] = data_all_2[sub][0: 8]
        data_all_new_2[sub][433:] = data_all_2[sub][16: 26]
        data_all_new_2[sub][207:215] = data_all_2[sub][8:16]
        data_all_new_2[sub][215:225] = data_all_2[sub][26:36]
        data_all_new_2[sub][0:207] = data_all_2[sub][36: 243]
        data_all_new_2[sub][225: 425] = data_all_2[sub][243:]
        data_all_new_2[sub][:, 425: 433] = data_all_2[sub][:, 0: 8]
        data_all_new_2[sub][:, 433:] = data_all_2[sub][:, 16: 26]
        data_all_new_2[sub][:, 207:215] = data_all_2[sub][:, 8:16]
        data_all_new_2[sub][:, 215:225] = data_all_2[sub][:, 26:36]
        data_all_new_2[sub][:, 0:207] = data_all_2[sub][:, 36: 243]
        data_all_new_2[sub][:, 225: 425] = data_all_2[sub][:, 243:]




    # 1 3
    # 2 4
    # 1和3 放到最后
    # 2和4 放到之前的11后面
    # data_all_new = np.zeros_like(data_all)
    # new_index = 0
    # culmulative = 0
    # for sub in data_all.shape[0]:
    #     for idx, set in enumerate(sett):
    #         if idx <= 3:
    #             culmulative += set
    #             continue
    #         if idx == 10:
    #             data_all_new[sub][new_index] = data_all[sub][idx]
    #             new_index += 1
    #             data_all_new[sub][new_index] = data_all[sub][1]
    #             new_index += 1
    #             data_all_new[sub][new_index] = data_all[sub][3]
    #             new_index +=1
    #         elif idx == 18:
    #         else:
    #             start_idx = culmulative
    #             end_idx = culmulative + set
    #             new_idx_start = new_index
    #             new_idx_end = new_index + set
    #             for i in range(data_all_new[sub].shape[0]):
    #                 data_all_new[sub][new_idx_start : new_idx_end] = data_all[sub][start_idx: end_idx]


    np.save(path1, data_all)
    path2 = "../fmri_precision_output/health_pre_before_precision_" + str(gamma) + ".npy"
    print("health pre文件保存在(无precision)：", path2)
    np.save(path2, data_all_2)

    # file = open("../fmri_precision_output/health_post_before_precision.txt", "w")
    # for sub in range(data_all.shape[0]):
    #     num = str(sub+1) + '\n'
    #     file.writelines(num)
    #     for i in range(data_all[sub].shape[0]):
    #         file.writelines(str(data_all[sub][i]) + '\n')
    #
    # file = open("../fmri_precision_output/health_pre_before_precision.txt", "w")
    # for sub in range(data_all_2.shape[0]):
    #     num = str(sub+1) + '\n'
    #     file.writelines(num)
    #     for i in range(data_all_2[sub].shape[0]):
    #         file.writelines(str(data_all_2[sub][i]) + '\n')

    # data_all = precision(data_all, health_num, dim, gamma)
    # data_all_2 = precision(data_all_2, health_num, dim, gamma)
    # path = "../fmri_precision_output/health_post_after_precision_" + str(gamma) + "_.txt"
    #
    # file = open(path, "w")
    # for sub in range(data_all.shape[0]):
    #     num = str(sub + 1) + '\n'
    #     file.writelines(num)
    #     for i in range(data_all[sub].shape[0]):
    #         file.writelines(str(data_all[sub][i]) + '\n')
    #
    # path = "../fmri_precision_output/health_pre_after_precision_" + str(gamma) + "_.txt"
    #
    # file = open(path, "w")
    # for sub in range(data_all_2.shape[0]):
    #     num = str(sub + 1) + '\n'
    #     file.writelines(num)
    #     for i in range(data_all_2[sub].shape[0]):
    #         file.writelines(str(data_all_2[sub][i]) + '\n')
    #
    # path = "../fmri_precision_output/health_post_after_precision_" + str(gamma) + "_.npy"
    #
    # np.save(path, data_all)
    # path = "../fmri_precision_output/health_pre_after_precision_" + str(gamma) + "_.npy"
    # np.save(path, data_all_2)
    #


    print("finish job, save health origin data")
    return path1, path2

if __name__ == "__main__":
    run("gamma")