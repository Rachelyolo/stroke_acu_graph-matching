import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from matplotlib import pyplot as plt
def run(gamma,path1, path2):
    dict_sample_s = {"data":[], "type":[], "x":[]}
    # path = "../fmri_costmap_output/before_costmap_stroke_" + str(gamma) + ".npy"
    print("正在进行通过清0矩阵将stroke中的矩阵进行去噪的工作")
    cost_map_all_s = np.load(path1)
    print("读取stroke costmap：", path1)
    # path = "flag_matrix_fmri_" + str(gamma) + ".npy"

    fmri_flag = np.load(path2)
    print("读取清0标记矩阵：", path2)
    # for i in range(fmri_flag.shape[0]):
    #     for j in range(fmri_flag.shape[1]):
    #         if fmri_flag[i][j] > 0:
    #             lk += 1
    # print(lk)
    # input()
    dim = cost_map_all_s.shape[1]
    data_frame_s = pd.DataFrame(dict_sample_s)
    health_num = cost_map_all_s.shape[0]
    transform_matrix_s = np.zeros([cost_map_all_s.shape[0], cost_map_all_s.shape[1], cost_map_all_s.shape[2]])
    head_num = 16
    network_trans = np.zeros([head_num, head_num])
    network_trans_num = np.zeros([head_num, head_num])
    samplex = []
    sampley = []
    num_count = 0
    people = np.zeros([cost_map_all_s.shape[0], 1])
    ss = []
    yy = []
    for sub in range(cost_map_all_s.shape[0]): # 第i个病人
        x_sample = []
        y_sample = []
        num_count = 0
        cost = cost_map_all_s[sub]
        rowind, colind = linear_sum_assignment(cost)
        for index in rowind:
            if index == colind[index]:
                transform_matrix_s[sub][index, colind[index]] = 1
            else:
                x_sample.append(index)
                y_sample.append(colind[index])

                transform_matrix_s[sub][index, colind[index]] = 1
        sub_list = [str(sub)] * dim
        dict_sample_s['data'].extend(y_sample)
        dict_sample_s['x'].extend(x_sample)
        dict_sample_s['type'].extend(sub_list)
        # people[sub] = num_count
        samplex.extend(x_sample)
        sampley.extend(y_sample)
        plt.figure()
        plt.scatter(x_sample, y_sample, s=8)
        ax = plt.gca()
        ax.invert_yaxis()
        plt.savefig("1111.png")
        print("dfdgdg")
        input()
    print("------output heatmap--------")

    sample_x_new = []
    for x in samplex:
        if x >= 0 and x < 8:
            sample_x_new.append(x + 425 - 7)
        elif x < 26 and x <= 16:
            sample_x_new.append(x + 417 - 7)
        elif x >= 8 and x < 16:
            sample_x_new.append(x + 199 - 7)
        elif x >= 26 and x < 36:
            sample_x_new.append(x + 189 - 7)
        elif x >= 43 and x < 243:
            sample_x_new.append(x - 36 - 7)
        elif x >= 243:
            sample_x_new.append(x - 18 - 7)
        # else:
        #     sample_x_new.append(x)

    sample_y_new = []
    for x in sampley:
        if x >= 0 and x < 8:
            sample_y_new.append(x + 425 - 7)
        elif x < 26 and x <= 16:
            sample_y_new.append(x + 417 - 7)
        elif x >= 8 and x < 16:
            sample_y_new.append(x + 199 - 7)
        elif x >= 26 and x < 36:
            sample_y_new.append(x + 189 - 7)
        elif x >= 43 and x < 243:
            sample_y_new.append(x - 36 - 7)
        elif x >= 243:
            sample_y_new.append(x - 18 - 7)

    plt.figure()
    ax = plt.gca()
    # ax.set_facecolor('#FFE6C7')
    # fig, ax = plt.subplots()
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    plt.scatter(sample_x_new, sample_y_new, s=0.1, alpha=0.2)
    x = np.arange(0, 436, 1)
    yy = np.arange(0, 436, 1)
    sett = [30, 40, 23, 25, 13, 30, 39, 18, 30, 40, 23, 25, 13, 30, 39, 18]
    x_tick = ['VIS_R', 'SOM_R', 'DA_R', 'VA_R', 'LIM_R',
              'FPN_R',
              'DMN_R', 'SC/CB_R', 'VIS_L', 'SOM_L', 'DA_L', 'VA_L', 'LIM_L',
              'FPN_L',
              'DMN_L', 'SC/CB_L']

    position = []
    start = 0
    for i in sett:
        ii = i // 2 + start
        position.append(ii)
        start += i

    print(x)
    start = 0
    for i in sett:
        start += i
        y = [start] * len(x)
        plt.plot(x, y, c="gray", linewidth=0.5)
        # xx = [start] * len(x)
        plt.plot(y, yy, c="gray", linewidth=0.5)

    plt.xlim((0, 435))
    plt.ylim((0, 435))
    plt.yticks(position, x_tick, size=7)
    plt.xticks(position, x_tick, rotation=90, size=7)

    # plt.plot(sample_x_new, )
    ax.invert_yaxis()
    # plt.show()
    plt.savefig("sky_fmri.png")
    input()


    data_frame_s = pd.DataFrame(dict_sample_s)


    sample_y_all = []
    temp = 0
    transform_matrix_all = np.sum(transform_matrix_s, axis=0)
    manto_save = np.zeros([cost_map_all_s.shape[0], head_num])
    for sub in range(transform_matrix_s.shape[0]):
        for i in range(transform_matrix_s.shape[1]):
            for j in range(transform_matrix_s.shape[2]):
                if i == j or i == j - 1 or i == j+1:
                    transform_matrix_s[sub][i, j] = 0
                if fmri_flag[i, j] == 1:
                    transform_matrix_s[sub][i, j] = 0
    for sub in range(transform_matrix_s.shape[0]):
        num_count = 0
        for i in range(transform_matrix_s.shape[1]):
            for j in range(transform_matrix_s.shape[2]):
                if transform_matrix_s[sub][i, j] == 1:
                    num_count +=1
        people[sub] = num_count
    people = pd.DataFrame(people)

    path = "../num/fmri_stroke_num_sub_" + str(gamma) + ".xlsx"
    print("将stroke每个人的变化次数保存到：", path)
    people.to_excel(path)

    transform_matrix_all = np.sum(transform_matrix_s, axis=0)
    sett = [8, 8, 10, 10, 7, 30, 40, 23, 25, 13, 30, 39, 30, 40, 23, 25, 13, 30, 39]
    # sett = [7, 30, 40, 23, 25, 13, 30, 39, 8, 10, 30, 40, 23, 25, 13, 30, 39, 8, 10]
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

    network_load_new = {}
    for key in network_load.keys():
        if key <= 4:
            continue
        network_load_new[key] = network_load[key]
    network_load_new[1] = network_load[1]
    network_load_new[1].extend(network_load[3])
    network_load_new[2] = network_load[2]
    network_load_new[2].extend(network_load[4])

    manto_save = np.zeros([cost_map_all_s.shape[0], head_num])
    for i in range(transform_matrix_s.shape[0]):
        network_trans = np.zeros([head_num, head_num])
        network_trans_num = np.zeros([head_num, head_num])
        x_tick = ['5', '6', '7', '8', '9', '10', '11', '2', '12', '13', '14', '15', '16', '17', '18', '1']
        # y_tick = ['5', '6', '7', '8', '9', '10', '11', '2', '4', '12', '13', '14', '15', '16', '17', '18', '1', '3']
        transform_matrix_all = transform_matrix_s[i]
        for idx, set_order in enumerate(x_tick):
            current_network = idx
            network_dim = network_load[int(set_order)]

            for small_dim_curr in network_dim:
                for idx2, set_order2 in enumerate(x_tick):
                    change_network = idx2
                    network_dum = network_load[int(set_order2)]
                    for small_dim_change in network_dum:
                        # if small_dim_change == small_dim_curr:
                        #     continue
                        network_trans[current_network, change_network] += transform_matrix_all[small_dim_curr][
                                                                              small_dim_change] / (
                                                                              len(network_dim)
                                                                          )
                        network_trans_num[current_network, change_network] += transform_matrix_all[small_dim_curr][
                            small_dim_change]
        # print(network_trans_num.shape)
        nnn = []
        for ii in range(network_trans_num.shape[0]):
            total = np.sum(network_trans_num[ii])
            if total == 0:
                nnn.append(1.0)
            else:
                nnn.append((total - network_trans_num[ii, ii]) / total)
        manto_save[i] = nnn
        network_trans_save = pd.DataFrame(network_trans)
        path = "../csv_output_fmri_stroke/"
        if i < 10:
            network_trans_save.to_csv(path + "0" + str(i) + "fmri_stroke_sub.csv")
        else:
            network_trans_save.to_csv(path + str(i) + "fmri_stroke_sub.csv")

    manto_save = pd.DataFrame(manto_save)
    manto_save.to_excel("../yblovesmanto/fmri_stroke.xlsx")

    # x_tick=['1','2','3','4','5','6','7','8','10','11','12','13','14','15','16','17']
    x_tick = ['5', '6', '7', '8', '9', '10', '11', '2', '12', '13', '14', '15', '16', '17', '18', '1']
    # transform_matrix_s
    # change_record = np.zeros([436, 1])
    # for i in range(7, transform_matrix_s[0].shape[0]):
    #
    #     base = 0
    #     for sub in range(transform_matrix_s.shape[0]):
    #         for j in range(7, transform_matrix_s[0].shape[1]):
    #             if transform_matrix_s[sub][i, j] == 1:
    #                 pair = [i, j]
    #                 for pairr in network_load_new.values():
    #                     if set(pair).issubset(set(pairr)):
    #                         base += 1
    #                         break
    #     base = base / transform_matrix_s.shape[0]
    #     change_record[i-7, 0] = base
    # # print(change_record.shape)
    # path = "../num/fmri_stroke_naoqu_sub_" + str(gamma) + ".xlsx"
    # print("将fmri stroke每个脑区的变化次数保存到：", path)
    # # people.to_excel(path)
    # change_record = pd.DataFrame(change_record)
    # change_record.to_excel(path)






    # y_tick = ['5', '6', '7', '8', '9', '10', '11', '2', '4', '12', '13', '14', '15', '16', '17', '18', '1', '3']

    network_trans = np.zeros([head_num, head_num])
    network_trans_num = np.zeros([head_num, head_num])
    transform_matrix_all = np.sum(transform_matrix_s, axis=0)
    num_record = np.zeros([436, 436])
    iss = 0
    for idx, set_order in enumerate(x_tick):
        jss = 0
        current_network = idx
        network_dim = network_load_new[int(set_order)]
        # print(network_dim)
        for small_dim_curr in network_dim:
            jss = 0
            for idx2, set_order2 in enumerate(x_tick):

                change_network = idx2
                network_dum = network_load_new[int(set_order2)]
                for small_dim_change in network_dum:
                    # num_record[iss, jss] += transform_matrix_all[small_dim_curr][
                    #                             small_dim_change] / transform_matrix_s.shape[0]
                    if small_dim_change == small_dim_curr:
                        continue
                    if current_network == change_network:
                        num_record[iss, jss] += transform_matrix_all[small_dim_curr][
                                                small_dim_change] / transform_matrix_s.shape[0]



                    network_trans[current_network, change_network] += transform_matrix_all[small_dim_curr][
                                                                          small_dim_change] / (
                                                                              len(network_dim) *
                                                                              transform_matrix_s.shape[0])

                    network_trans_num[current_network, change_network] += transform_matrix_all[small_dim_curr][
                        small_dim_change]
                    jss+=1
            iss += 1

    # print(num_record)

    # data_all_new = np.zeros([transform_matrix_s.shape[0], 443, 443])
    # data_all = transform_matrix_s
    #
    # l
    # for sub in range(transform_matrix_s.shape[0]):
    #     data_all_new[sub][425: 433] = data_all[sub][0: 8]
    #     data_all_new[sub][433:] = data_all[sub][16: 26]
    #     data_all_new[sub][207:215] = data_all[sub][8:16]
    #     data_all_new[sub][215:225] = data_all[sub][26:36]
    #     data_all_new[sub][0:207] = data_all[sub][36: 243]
    #     data_all_new[sub][225: 425] = data_all[sub][243:]
    #     data_all_new[sub][:, 425: 433] = data_all[sub][:, 0: 8]
    #     data_all_new[sub][:, 433:] = data_all[sub][:, 16: 26]
    #     data_all_new[sub][:, 207:215] = data_all[sub][:, 8:16]
    #     data_al_new[sub][:, 215:225] = data_all[sub][:, 26:36]
    #     data_all_new[sub][:, 0:207] = data_all[sub][:, 36: 243]
    #     data_all_new[sub][:, 225: 425] = data_all[sub][:, 243:]
    # iss = 0
    # jss = 0
    # # for sub in range(transform_matrix_s.shape[0]):
    # #     iss = 0
    # #     jss = 0
    # #     for idx, set_order in enumerate(x_tick):
    # #         current_network = idx
    # #         network_dim = network_load_new[int(set_order)]
    # #         # print(network_dim)
    # #         for small_dim_curr in network_dim:
    # #             jss = 0
    # #             for idx2, set_order2 in enumerate(x_tick):
    # #                 change_network = idx2
    # #                 network_dum = network_load_new[int(set_order2)]
    # #                 for small_dim_change in network_dum:
    # #                     num_record[sub][iss, jss] = transform_matrix_s[sub][small_dim_curr][small_dim_change]
    # #                     jss +=1
    # #             iss+=1
    #
    # # transform_matrix_s
    # num_record = transform_matrix_s
    # num_record = np.sum(num_record, axis=0)
    change_record = np.zeros([436, 1])
    for i in range(num_record.shape[0]):
        base = 0
        for j in range(num_record.shape[1]):
            base += num_record[i, j]

        change_record[i, 0] = base
    # print(change_record.shape)
    path = "../num/fmri_stroke_naoqu_sub_" + str(gamma) + ".xlsx"
    print("将fmri stroke每个脑区的变化次数保存到：", path)
    # people.to_excel(path)
    change_record = pd.DataFrame(change_record)
    change_record.to_excel(path)


    # print(network_trans)
    # np.save("../threold_research/fmri_stroke.npy", network_trans)

    # np.save("../fmri_heatmap_output/fmri_before_stroke_network_Z.npy", network_trans)

    x_tick = ['VIS_R', 'SOM_R', 'DA_R', 'VA_R', 'LIM_R',
              'FPN_R',
              'DMN_R', 'SC/CB_R', 'VIS_L', 'SOM_L', 'DA_L', 'VA_L', 'LIM_L',
              'FPN_L',
              'DMN_L', 'SC/CB_L']
    y_tick = x_tick
    for idx, i in enumerate(x_tick):
        x_tick[idx] = i.replace(' ', '-')
    for idx, j in enumerate(y_tick):
        y_tick[idx] = j.replace(' ', '-')

    data = {}

    for i in range(16):
        data[x_tick[i]] = network_trans[i]
    plt.figure(figsize=(12, 11))
    import seaborn as sns

    pd_data = pd.DataFrame(data, index=y_tick, columns=x_tick)
    sns.heatmap(data=pd_data, square=True, annot=True)
    path = "../fmri_heatmap_output/fmri_before_stroke_heatmap_percent_" + str(gamma) + ".png"
    print("将去噪后的stroke的heatmap frequency图保存到：", path)
    plt.savefig(path)

    data = {}

    for i in range(16):
        data[x_tick[i]] = network_trans_num[i]
    plt.figure(figsize=(12, 11))

    pd_data = pd.DataFrame(data, index=y_tick, columns=x_tick)
    sns.heatmap(data=pd_data, square=True, annot=True)

    path = "../fmri_heatmap_output/fmri_before_stroke_heatmap_num_" + str(gamma) + ".png"
    print("将去噪后的stroke的heatmap number图保存到：", path)

    plt.savefig(path)
# if __name__ == "__main__":
#     run()

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
    #
    #
    # for i in range(transform_matrix_s.shape[0]):
    #     network_trans = np.zeros([head_num, head_num])
    #     network_trans_num = np.zeros([head_num, head_num])
    #     temp_ls = 0
    #     mm = 0
    #
    #     x_tick = ['5', '6', '7', '8', '9', '10', '11', '2', '4', '12', '13', '14', '15', '16', '17', '18', '1', '3']
    #     y_tick = ['5', '6', '7', '8', '9', '10', '11', '2', '4', '12', '13', '14', '15', '16', '17', '18', '1', '3']
    #     transform_matrix_all = transform_matrix_s[i]
    #     for idx, set_order in enumerate(x_tick):
    #         current_network = idx
    #         network_dim = network_load[int(set_order)]
    #         for small_dim_curr in network_dim:
    #             for idx2, set_order2 in enumerate(x_tick):
    #                 change_network = idx2
    #                 network_dum = network_load[int(set_order2)]
    #                 for small_dim_change in network_dum:
    #                     # if small_dim_change == small_dim_curr:
    #                     #     continue
    #                     network_trans[current_network, change_network] += transform_matrix_all[small_dim_curr][
    #                                                                           small_dim_change] / (
    #                                                                               len(network_dim) )
    #                     network_trans_num[current_network, change_network] += transform_matrix_all[small_dim_curr][
    #                         small_dim_change]
    #
    #     nnn = []
    #     for ii in range(network_trans_num.shape[0]):
    #         total = np.sum(network_trans_num[ii])
    #         if total == 0:
    #             nnn.append(1.0)
    #         else:
    #             nnn.append((total - network_trans_num[ii, ii]) / total)
    #     manto_save[i] = nnn
    #     network_trans_save = pd.DataFrame(network_trans)
    #     path = "../csv_output_fmri_stroke/"
    #     if i<10:
    #         network_trans_save.to_csv(path + "0"+str(i) +"fmri_stroke_sub_precision.csv")
    #     else:
    #         network_trans_save.to_csv(path + str(i) + "fmri_stroke_sub_precision.csv")
    # manto_save = pd.DataFrame(manto_save)
    # manto_save.to_excel("../yblovesmanto/fmri_stroke_precision.xlsx")
    #
    # #--------- all people------------------
    # network_trans = np.zeros([head_num, head_num])
    # network_trans_num = np.zeros([head_num, head_num])
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
    #
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
    #                 # if small_dim_change == small_dim_curr:
    #                 #     continue
    #                 network_trans[current_network, change_network] += transform_matrix_all[small_dim_curr][small_dim_change] / (
    #                             len(network_dim) * transform_matrix_s.shape[0])
    #                 network_trans_num[current_network, change_network] += transform_matrix_all[small_dim_curr][small_dim_change]
    # # print(network_trans)
    # # np.save("../threold_research/fmri_stroke.npy", network_trans)
    # #
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
    #
    # for i in range(18):
    #     data[x_tick[i]] = network_trans[i]
    # plt.figure(figsize=(12, 11))
    # import seaborn as sns
    #
    # pd_data = pd.DataFrame(data, index=y_tick, columns=x_tick)
    # sns.heatmap(data=pd_data, square=True)
    # plt.savefig("../fmri_heatmap_output/fmri_after_stroke_heatmap_percent_sub.png")
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
    # plt.savefig("../fmri_heatmap_output/fmri_after_stroke_heatmap_num_sub.png")
    # #
    # #
if __name__ == "__main__":
    run("fmri_v3", "../fmri_costmap_output/before_costmap_stroke_fmri_v3.npy", "flag_matrix_fmri_fmri_v3.npy")
