





import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("python-code/FC_self.csv")
# print(data)
import numpy as np
NSA = data["NSA"].to_numpy()
total = data['total'].to_numpy()

R = data["R"].to_numpy()
L = data['L'].to_numpy()
R = np.delete(R, 3)
L = np.delete(L, 3)
label_R = ['R'] * len(R)
label_L = ['L'] * len(L)
NSA_ = np.delete(NSA, 3)
NSA_ = NSA_.tolist()
R = R.tolist()

# R rs=0.4915202 P=0.01258
# L rs=0.4414 P=0.02718
L = L.tolist()
NSA_.extend(NSA_)
R.extend(L)
label_R.extend(label_L)
print(label_R)
# plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
# plt.rcParams["axes.unicode_minus"]=False
dict = {r'$\Delta$NSA': NSA, "remapping frequency": total}


dict1 = {r'$\Delta$NSA': NSA_, "remapping frequency":R , 'label': label_R}

# dict2 = {r'$\Delta$NSA': NSA_, "R/L":L , "label": label_L}
dict = pd.DataFrame(dict)
dict1 = pd.DataFrame(dict1)
# dict2 = pd.DataFrame(dict2)
# print(NSA)
print(dict)
plt.figure()
ax = plt.gca()
# sns.regplot(x="remapping frequency", y=r'$\Delta$NSA', data=dict, scatter_kws={'s':10}, ci=85)
sns.lmplot(x="remapping frequency", y=r'$\Delta$NSA', data=dict, scatter_kws={'s':10}, ci=85)
# plt.savefig("plot1.png")
plt.show()

# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)

plt.figure()

sns.lmplot(x="remapping frequency", y=r'$\Delta$NSA', data=dict1, scatter_kws={'s':10}, ci=85, hue='label', markers=['^','o'])
# sns.regplot(x="R/L", y=r'$\Delta$NSA', data=dict1, scatter_kws={'s':10}, ci=85, label='label')
# plt.legend(labels=["hhh"])
# sns.regplot(x="L", y=r'$\Delta$NSA', data=dict2, scatter_kws={'s':10}, ci=85)
# plt.legend(labels=["hhh, hhhs"])
plt.text(0.08, 0.05, r'$r_s$=0.4915 P=0.0126')
plt.text(0.08, 0.2, r'$r_s$=0.4414 P=0.0272')
plt.show()
