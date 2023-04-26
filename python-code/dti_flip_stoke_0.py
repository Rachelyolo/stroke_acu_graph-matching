import pandas as pd
import numpy as np
import os
def run(gamma):
    father_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "443")
    father_path = os.path.join(father_path, "connectome_dti_443")
    t0 = "stroke-post"# lack 19
    t1 = "stroke-pre" # lack 19
    dim = 443
    head_num = 18 # 脑区个数
    health_num = 26 - 1 # 人数

    flip_ = ["01","02","11","14","16","18","20","21","23","24","25","27"]
    # flip_ = []

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


    def flip(p, transform_matrix, change_transform, x1start, x1end, y1start, y1end):
        gapx = (x1end-x1start) // 2
        gapy = (y1end - y1start) // 2
        for i in range(x1start, x1start + gapx):
            for j in range(y1start, y1start + gapy):
                # print(transform_matrix[p][i + gapx, j + gapy])
                change_transform[p][i, j] = transform_matrix[p][i + gapx, j + gapy]
        # print(change_transform[p])
        for i in range(x1start + gapx, x1end):
            for j in range(y1start + gapy, y1end):
                change_transform[p][i, j] = transform_matrix[p][i - gapx, j - gapy]
        # print(change_transform[p])

        for i in range(x1start + gapx, x1end):
            for j in range(y1start, y1start + gapy):
                change_transform[p][i, j] = transform_matrix[p][i - gapx, j + gapy]
        # print(change_transform[p])
        for i in range(x1start, x1start + gapx):
            for j in range(y1start + gapy, y1end):
                change_transform[p][i, j] = transform_matrix[p][i + gapx, j - gapy]
        # print(change_transform[0])
        return change_transform

    sett = [8, 8, 10, 10, 7, 30, 40, 23, 25, 13, 30, 39, 30, 40, 23, 25, 13, 30, 39]
    # 1-8；9-16；17-26；27-36；44-73；74-113；114-136；137-161；162-174；175-204；205-243
    network_trans = np.zeros([head_num, head_num])

    time_0_path_health = os.path.join(father_path, t0)
    time_1_path_health = os.path.join(father_path, t1)

    from scipy.spatial import distance

    data_all = []
    flip_or_not = []
    for path, _, files in os.walk(time_0_path_health):
        files.sort(key=lambda x: int(x[14:16]))
        for idx, file in enumerate(files):
            file_name = os.path.join(path, file)
            if "19" in file or "02" in file:
                continue
            data_1 = pd.read_csv(file_name, header=None)
            data_1 = data_1.to_numpy()
            data = data_1
            df = 1
            for sss in flip_:
                if "sub" + sss in file:
                    flip_or_not.append(1)
                    df = 0
                    break
            if df == 1:
                flip_or_not.append(0)
            data_all.append(data)
    data_all = np.stack(data_all, axis=0)
    data_all_2 = []
    for path, _, files in os.walk(time_1_path_health):
        files.sort(key=lambda x: int(x[14:16]))
        for idx, file in enumerate(files):
            file_name = os.path.join(path, file)
            if "19" in file or "02" in file:
                continue
            data_1 = pd.read_csv(file_name, header=None)
            data_1 = data_1.to_numpy()
            data = data_1
            data_all_2.append(data)

    data_all_2 = np.stack(data_all_2, axis=0)
    cost_map_all = np.zeros([health_num, dim, dim])
    data_all_2[np.isnan(data_all_2)] = 0
    data_all_2[np.isinf(data_all_2)] = 0
    data_all[np.isnan(data_all)] = 0
    data_all[np.isinf(data_all)] = 0
    transform_matrix = data_all
    change_transform = np.zeros([health_num, dim, dim])
    print("读取的文件夹: ",time_0_path_health, " 以及 ", time_1_path_health)
    print("读取的人数: ", health_num)

    #
    # file = open("../dti_flip_output/dti_stroke_post_before_flip.txt", "w")
    # for sub in range(data_all.shape[0]):
    #     num = str(sub+1) + '\n'
    #     file.writelines(num)
    #     for i in range(data_all[sub].shape[0]):
    #         file.writelines(str(data_all[sub][i]) + '\n')
    #
    # file = open("../dti_flip_output/dti_stroke_pre_before_flip.txt", "w")
    # for sub in range(data_all_2.shape[0]):
    #     num = str(sub+1) + '\n'
    #     file.writelines(num)
    #     for i in range(data_all_2[sub].shape[0]):
    #         file.writelines(str(data_all_2[sub][i]) + '\n')

    # print(flip_or_not)
    if "dti" in father_path:
        for p in range(health_num):
            if flip_or_not[p] == 0:
                change_transform[p] = transform_matrix[p]
                continue
            change_transform = flip(p, transform_matrix, change_transform, 0, 16, 0, 16)
            change_transform = flip(p, transform_matrix, change_transform, 16, 36, 16, 36)

            change_transform = flip(p, transform_matrix, change_transform, 43, 443, 43, 443)

            change_transform = flip(p, transform_matrix, change_transform, 0, 16, 16, 36)

            change_transform = flip(p, transform_matrix, change_transform, 16, 36, 0, 16)

            change_transform = flip(p, transform_matrix, change_transform, 43, 443, 16, 36)
            change_transform = flip(p, transform_matrix, change_transform, 16, 36, 43, 443)
            change_transform = flip(p, transform_matrix, change_transform, 0, 16, 43, 443)
            change_transform = flip(p, transform_matrix, change_transform, 43, 443, 0, 16)
    else:
        change_transform = transform_matrix
    data_all = change_transform
    transform_matrix = data_all_2
    change_transform = np.zeros([health_num, dim, dim])
    if "dti" in father_path:
        for p in range(health_num):
            if flip_or_not[p] == 0:
                change_transform[p] = transform_matrix[p]
                continue
            change_transform = flip(p, transform_matrix, change_transform, 0, 16, 0, 16)
            change_transform = flip(p, transform_matrix, change_transform, 16, 36, 16, 36)
            change_transform = flip(p, transform_matrix, change_transform, 43, 443, 43, 443)
            change_transform = flip(p, transform_matrix, change_transform, 0, 16, 16, 36)
            change_transform = flip(p, transform_matrix, change_transform, 16, 36, 0, 16)
            change_transform = flip(p, transform_matrix, change_transform, 43, 443, 16, 36)
            change_transform = flip(p, transform_matrix, change_transform, 16, 36, 43, 443)
            change_transform = flip(p, transform_matrix, change_transform, 0, 16, 43, 443)
            change_transform = flip(p, transform_matrix, change_transform, 43, 443, 0, 16)
    else:
        change_transform = transform_matrix
    data_all_2 = change_transform

    #
    # file_name = "../dti_flip_output/dti_stroke_post_after_flip_"
    # for sub in range(data_all.shape[0]):
    #     file_name_ = file_name + str(sub+1) + ".txt"
    #     num = str(sub + 1) + '\n'
    #     file = open(file_name_, "w")
    #     # print(file_name_)
    #     file.writelines(num)
    #     for i in range(data_all[sub].shape[0]):
    #         file.writelines(str(data_all[sub][i]) + '\n')
    # file_name = "../dti_flip_output/dti_stroke_pre_after_flip_"
    # for sub in range(data_all_2.shape[0]):
    #     num = str(sub + 1) + '\n'
    #     file_name_ = file_name + str(sub+1) + ".txt"
    #     file = open(file_name_, "w")
    #     file.writelines(num)
    #     for i in range(data_all_2[sub].shape[0]):
    #         file.writelines(str(data_all_2[sub][i]) + '\n')

    data_all_new = np.zeros_like(data_all)
    for sub in range(data_all.shape[0]):
        data_all_new[sub][425: 433] = data_all[sub][0: 8]
        data_all_new[sub][433:] = data_all[sub][16: 26]
        data_all_new[sub][207:215] = data_all[sub][8:16]
        data_all_new[sub][215:225] = data_all[sub][26:36]
        data_all_new[sub][0:207] = data_all[sub][36: 243]
        data_all_new[sub][225: 425] = data_all[sub][243:]
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
    path_post = "../dti_flip_output/dti_stroke_post_after_flip_" + str(gamma) + ".npy"
    print("翻转后的stroke post的结果存储在了: ", path_post)
    np.save(path_post, data_all)
    path_pre = "../dti_flip_output/dti_stroke_pre_after_flip_" + str(gamma) + ".npy"
    print("翻转后的stroke pre的结果存储在了: ", path_pre)
    np.save(path_pre, data_all_2)

    print("翻转后的dti stroke结果存储完毕")
    return path_pre, path_post

# np.save("../dti_flip_output/dti_stroke_post_after_flip.npy", data_all)
# np.save("../dti_flip_output/dti_stroke_pre_after_flip.npy", data_all_2)

# print("finish job, save origin data")