#
import numpy as np
from scipy.spatial import distance


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

# dti_flip_health_post.npy
def run(gamma, path1, path2):
    data_all = np.load(path2)
    data_all_2 = np.load(path1)

    health_num = data_all.shape[0]
    dim = data_all.shape[1]
    cost_map_all = np.zeros([health_num, dim, dim])
    for subject_x in range(data_all.shape[0]):
        costmap = calculate_cost(data_all[subject_x], data_all_2[subject_x])
        cost_map_all[subject_x] = costmap

    # file = open("../dti_costmap_output/dti_flip_health_costmap.txt", "w")
    # for sub in range(cost_map_all.shape[0]):
    #     num = str(sub+1) + '\n'
    #     file.writelines(num)
    #     for i in range(cost_map_all[sub].shape[0]):
    #         file.writelines(str(cost_map_all[sub][i]) + '\n')
    path = "../dti_costmap_output/dti_flip_health_costmap_" + str(gamma) + ".npy"
    np.save(path, cost_map_all)
    print("health costmap存储在了: ", path)
    print("finish save costmap")
    return path