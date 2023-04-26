import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
plt.figure()
sns.set(font_scale=1.5)

# data=sns.load_dataset("flights")\
        # .pivot("month","year","passengers")
# data.head()
# print(data)
# sns.heatmap(data=data,vmin=200,vmax=500)
# plt.show()

import numpy as np

x1 = np.zeros([16, 16])
pair = [[10, 0], [12, 2], [14, 16]]
for i in range(x1.shape[0]):
    for j in range(x1.shape[1]):
        if i == j:
            x1[i , j] = 0.2
start = 0
for i in range(8, 16):
    x1[i, start] = 0.2
    x1[start, i] = 0.2
    start +=1
for i in range(x1.shape[0]):
    for j in range(x1.shape[1]):
        if x1[i,j] == 0:

            x1[i, j] = np.random.random() / 10.0

n = pd.DataFrame(x1)
print(n)

x_tick = ['VIS_R', 'SOM_R', 'DA_R', 'VA_R', 'LIM_R',
              'FPN_R',
              'DMN_R', 'SC/CB_R', 'VIS_L', 'SOM_L', 'DA_L', 'VA_L', 'LIM_L',
              'FPN_L',
              'DMN_L', 'SC/CB_L']

dict = {}
for i in range(x1.shape[0]):
    dict[x_tick[i]] = x1[i]
n = pd.DataFrame(dict, index=x_tick, columns=x_tick)
print(n)
plt.figure(figsize=(12, 11))
sns.heatmap(n,vmax=0.25, square=True)
plt.xticks(fontsize=9) #x轴刻度的字体大小（文本包含在pd_data中了）
plt.yticks(fontsize=9)
# plt.show()
plt.savefig("sample.png")
