





import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("couple-selfR.csv")
# print(data)
import numpy as np
# NSA = data["NSA"].to_numpy()
# total = data['total'].to_numpy()
FC_F=data['fctotal'].to_numpy()
SC_F=data['sctotal'].to_numpy()
pair = []
num = 0
for i in range(len(FC_F)):
    print(i)
    if [FC_F[i], SC_F[i]] in pair:
        continue
    else:
        pair.append([FC_F[i], SC_F[i]])
        num+=1
print(num)
input()


# R = data["R"].to_numpy()
# L = data['L'].to_numpy()
# R = np.delete(R, 3)
# L = np.delete(L, 3)
# label_R = ['R'] * len(R)
# label_L = ['L'] * len(L)
# NSA_ = np.delete(NSA, 3)
# NSA_ = NSA_.tolist()
# R = R.tolist()


# L = L.tolist()
# NSA_.extend(NSA_)
# R.extend(L)
# label_R.extend(label_L)
# print(label_R)

# dict = {r'$\Delta$NSA': NSA, "remapping frequency": total}

# dict1 = {r'$\Delta$NSA': NSA_, "remapping frequency":R , 'label': label_R}
dict2={"FC_F":FC_F,"SC_F":SC_F}

# dict2 = {r'$\Delta$NSA': NSA_, "R/L":L , "label": label_L}
dict2 = pd.DataFrame(dict2)

# dict1 = pd.DataFrame(dict1)
# dict2 = pd.DataFrame(dict2)
# print(NSA)
print(dict2)
# print(dict2)
plt.figure(figsize=(12, 11))
# sns.lmplot(x="remapping frequency", y=r'$\Delta$NSA', data=dict, scatter_kws={'s':10}, ci=85)

sns.regplot(x="FC_F", y="SC_F", data=dict2, scatter_kws={'s': 8}, ci=85)
# sns.regplot(x="remapping frequency", y=r'$\Delta$NSA', data=dict, scatter_kws={'s':10}, ci=85)
# plt.text(0.08, 0.25, r'$r_s$=0.2930 P=0.4.379e-10')
plt.xlim(xmin=-0.005)
plt.ylim(ymin=-0.005)
plt.show()
# plt.savefig("FC-SC.png")

# plt.figure()
# sns.lmplot(x="remapping frequency", y=r'$\Delta$NSA', data=dict1, scatter_kws={'s':10}, ci=85, hue='label', markers=['^','o'])
# plt.text(0.08, 0.05, r'$r_s$=0.4071 P=0.0434')
# plt.text(0.08, 0.2, r'$r_s$=0.4915 P=0.0126')
# plt.savefig("FC-selfRL.png")
