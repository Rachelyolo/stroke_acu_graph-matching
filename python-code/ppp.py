import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.collections as clt
import ptitprince as pt

dti = pd.read_excel("../num/dti_final.xlsx").to_numpy() / 443.0
dti = dti[:, 1]
dti = dti.tolist()
# print(dti)
fmri = pd.read_excel("../num/fmri_final.xlsx").to_numpy() / 443.0
fmri = fmri[:, 1]
fmri = fmri.tolist()
# print(fmri)
g = ["SC"] * len(dti) + ['FC'] * len(fmri)
dti.extend(fmri)
dict = {"group": g, "remapping frequency":dti}
dict = pd.DataFrame(dict)
print(dict)
dx="group"; dy="remapping frequency"; ort="v"; pal="Set2"; sigma=.2
f, ax=plt.subplots(figsize=(7, 5))
ax=pt.RainCloud(x=dx, y=dy, data=dict, palette=pal, bw=sigma,
                 width_viol=.5, ax=ax, orient=ort)
# plt.title("Figure P10\n Flipping your Rainclouds")/
plt.savefig('figureP07.png', bbox_inches='tight')
# ax.show()
# plt.savefig("sssss.png")