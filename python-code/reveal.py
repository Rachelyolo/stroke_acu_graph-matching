# n = input("你要研究谁？example：stroke_health\n")
n = "fmri_health"
path = "../csv_output_" + n
import os

import numpy as np
import pandas as pd
# save = np.zeros()
for s, _, files in os.walk(path):
    files.remove('.DS_Store')
    files.sort(key=lambda x: int(x[0:2]))

    people = len(files)
    save = np.zeros([len(files), 18, 18])
    for index, file in enumerate(files):
        d = os.path.join(s, file)
        r = pd.read_csv(d).to_numpy()
        # print(r[:, 1:])
        save[index] = r[:, 1:]

points = [[1, 1], [2, 3]]


# output = []

for idx, point in enumerate(points):
    temp = []
    for j in range(save.shape[0]):
        num = save[j][point[0], point[1]]
        temp.append(num)
    temp = np.reshape(np.array(temp), [people, 1])
    # print(temp.shape)shape
    if idx == 0:
        output = temp
    else:
        output = np.concatenate([output, temp], axis=1)

output = pd.DataFrame(output)
path = "../csv_reveal/" + n +".xlsx"
output.to_excel(path)

#
# print(path)
# n = "all"
# # n = input("你要研究哪几个人？以空格来区分 如果输入all 代表所有人都要取\n")
# if n == "all":
#     s = [[1,1], [2,2], [3,3]]
#     for index in s:
#         for j in



