import numpy as np
def run(gamma, path1, path2):
    # dti_health = np.load("../threold_research/dti_health_node.npy")
    # dti_stroke = np.load("../threold_research/dti_stroke_node.npy")

    # path = "../threold_research/fmri_health_node_" + str(gamma) + ".npy"

    dti_health = np.load(path1)
    # path = "../threold_research/fmri_stroke_node_" + str(gamma) + ".npy"
    dti_stroke = np.load(path2)

    # fmri_stroke_test = np.load("../threold_research/fmri_stroke_node.npy")

    # for i in range(fmri_health.shape[0]):
    #     for j in range(fmri_health.shape[0]):
    #         if fmri_health[i,j] >= 7:
    #             print(fmri_stroke[i, j])
    # input()
    # fmri_stroke_test = fmri_stroke
    thr = 5
    print("正在进行通过阈值去噪的工作")
    print("阈值选取为: ", thr)


    # flag_matrix_dti = np.zeros_like(dti_health)
    flag_matrix_dti = np.zeros_like(dti_health)
    # print()

    # sett = [8, 8, 10, 10, 7, 30, 40, 23, 25, 13, 30, 39, 30, 40, 23, 25, 13, 30, 39]
    # for i in range(dti_health.shape[0]):
    #     for j in range(dti_health.shape[0]):
    #         if i == j:
    #             continue
    #         if dti_health[i, j] >= thr:
    #             flag_matrix_dti[i, j] = 1
    #             dti_stroke[i, j] = 0
            # if dti_stroke[i, j] < 2:
            #     dti_stroke[i, j] = 0
            #     flag_matrix_dti[i, j] = 1




    for i in range(dti_health.shape[0]):
        for j in range(dti_health.shape[0]):
            if i == j:
                continue
            if dti_health[i, j] >= thr:
                # print(fmri_stroke[i,j], "before")
                flag_matrix_dti[i, j] = 1
                dti_stroke[i, j] = 0
            # if fmri_stroke[i, j] < 2:
            #     flag_matrix_fmri[i, j] = 1
            #     fmri_stroke[i, j] = 0
            #     # print(fmri_stroke[i, j], "after")
    # thr = 3
    # print(np.count_nonzero(flag_matrix_dti))
    # input()
    # print(np.count_nonzero(flag_matrix_fmri))
    print(np.count_nonzero(flag_matrix_dti))
    pathmm = "flag_matrix_dti_" + str(gamma)+".npy"
    np.save(pathmm, flag_matrix_dti)
    print("标记哪些坐标清0的443*443矩阵存在: ", pathmm)


    # np.save("flag_matrix_dti.npy", flag_matrix_dti)
    #
    #
    # sett = [8, 8, 10, 10, 7, 30, 40, 23, 25, 13, 30, 39, 30, 40, 23, 25, 13, 30, 39]
    sett = [7, 30, 40, 23, 25, 13, 30, 39, 8, 10, 30, 40, 23, 25, 13, 30, 39, 8, 10]










    head_num = 16
    network_trans = np.zeros([head_num, head_num])
    network_trans_num = np.zeros([head_num, head_num])





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

    transform_matrix_all = dti_stroke
    temp_ls = 0
    mm =0

    x_tick = ['1', '2', '3', '4', '5', '6', '7', '8', '10', '11', '12', '13', '14', '15', '16', '17']

    # x_tick=['5','6','7','8','9','10','11','2','4','12','13','14','15','16','17','18','1','3']
    # y_tick=['5','6','7','8','9','10','11','2','4','12','13','14','15','16','17','18','1','3']

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
                    network_trans[current_network, change_network] += transform_matrix_all[small_dim_curr][small_dim_change] / (
                                len(network_dim) * 25.0)
                    network_trans_num[current_network, change_network] += transform_matrix_all[small_dim_curr][small_dim_change]
    # print(network_trans)
    # np.save("../threold_research/fmri_stroke.npy", network_trans)

    # np.save("../fmri_heatmap_output/fmri_before_stroke_network_Z.npy", network_trans)

    x_tick = ['R visual', 'R somatosensory', 'R dorsal attention', 'R ventral attention', 'R limbic', 'R fronto-parietal',
              'R default mode', 'R subcortical', 'L visual', 'L somatosensory', 'L dorsal attention',
              'L ventral attention', 'L limbic', 'L fronto-parietal', 'L default mode', 'L subcortical']
    y_tick = x_tick
    for idx, i in enumerate(x_tick):
        x_tick[idx] = i.replace(' ', '-')
    for idx, j in enumerate(y_tick):
        y_tick[idx] = j.replace(' ', '-')

    data = {}
    import matplotlib.pyplot as plt
    import pandas as pd
    for i in range(16):
        data[x_tick[i]] = network_trans[i]
    plt.figure(figsize=(12, 11))
    import seaborn as sns

    pd_data = pd.DataFrame(data, index=y_tick, columns=x_tick)
    sns.heatmap(data=pd_data, square=True)

    path = "../dti_heatmap_output/dti_stroke_heatmap_percent_sub_" + str(thr) + "_"+str(gamma)+ ".png"
    plt.savefig(path)
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
    #
    # path = "../fmri_heatmap_output/fmri_before_stroke_heatmap_num_Z_sub_" + str(thr) + ".png"
    # plt.savefig(path)
    #
    # head_num = 18
    # network_trans = np.zeros([head_num, head_num])
    # network_trans_num = np.zeros([head_num, head_num])
    #
    #
    #
    #
    #
    # network_load = {}
    # label = 1
    # start = 0
    # for i in sett:
    #     if i == 7:
    #         start += i
    #         continue
    #     for idx, j in enumerate(range(start, i+start)):
    #         if idx == 0:
    #             network_load[label] = [start]
    #         else:
    #             network_load[label].append(j)
    #     start += i
    #     label+=1
    #
    # transform_matrix_all = dti_stroke
    # temp_ls = 0
    # mm =0
    #
    # x_tick=['5','6','7','8','9','10','11','2','4','12','13','14','15','16','17','18','1','3']
    # y_tick=['5','6','7','8','9','10','11','2','4','12','13','14','15','16','17','18','1','3']
    #
    # for idx, set_order in enumerate(x_tick):
    #     current_network = idx
    #     network_dim = network_load[int(set_order)]
    #     for small_dim_curr in network_dim:
    #         for idx2, set_order2 in enumerate(x_tick):
    #             change_network = idx2
    #             network_dum = network_load[int(set_order2)]
    #             for small_dim_change in network_dum:
    #                 if small_dim_change == small_dim_curr:
    #                     continue
    #                 network_trans[current_network, change_network] += transform_matrix_all[small_dim_curr][small_dim_change] / (
    #                             len(network_dim) * 25.0)
    #                 network_trans_num[current_network, change_network] += transform_matrix_all[small_dim_curr][small_dim_change]
    # # print(network_trans)
    # # np.save("../threold_research/fmri_stroke.npy", network_trans)
    #
    # # np.save("../fmri_heatmap_output/fmri_before_stroke_network_Z.npy", network_trans)
    #
    # x_tick = ['R visual', 'R somatosensory', 'R dorsal attention', 'R ventral attention', 'R limbic', 'R fronto-parietal',
    #           'R default mode', 'R subcortical', 'R cerebellum', 'L visual', 'L somatosensory', 'L dorsal attention',
    #           'L ventral attention', 'L limbic', 'L fronto-parietal', 'L default mode', 'L subcortical', 'L cerebellum']
    # y_tick = ['R visual', 'R somatosensory', 'R dorsal attention', 'R ventral attention', 'R limbic', 'R fronto-parietal',
    #           'R default mode', 'R subcortical', 'R cerebellum', 'L visual', 'L somatosensory', 'L dorsal attention',
    #           'L ventral attention', 'L limbic', 'L fronto-parietal', 'L default mode', 'L subcortical', 'L cerebellum']
    # for idx, i in enumerate(x_tick):
    #     x_tick[idx] = i.replace(' ', '-')
    # for idx, j in enumerate(y_tick):
    #     y_tick[idx] = j.replace(' ', '-')
    #
    # data = {}
    # import matplotlib.pyplot as plt
    # import pandas as pd
    # for i in range(18):
    #     data[x_tick[i]] = network_trans[i]
    # plt.figure(figsize=(12, 11))
    # import seaborn as sns
    #
    # pd_data = pd.DataFrame(data, index=y_tick, columns=x_tick)
    # sns.heatmap(data=pd_data, square=True)
    # path = "../dti_heatmap_output/dti_before_stroke_heatmap_percent_Z_sub_" + str(thr) + ".png"
    # plt.savefig(path)
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
    # path = "../dti_heatmap_output/dti_before_stroke_heatmap_num_Z_sub_" + str(thr) + ".png"
    #
    # plt.savefig(path)
    return pathmm



