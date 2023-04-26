
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import pandas as pd
data = pd.read_csv("multiple.csv")
print(data)
label = data['NSA'].to_numpy()[0:-1]
fcf = data['FC_F'].to_numpy()[0:-1]
scf = data['SC_F'].to_numpy()[0:-1]
x1 = np.linspace(fcf.min(), fcf.max(), 50)
x2 = np.linspace(scf.min(), scf.max(), 50)
x1, x2 = np.meshgrid(x1, x2)
f = 0.46997 * x1 +  0.20146 * x2 - 0.18815

NSA = []
fig = plt.figure()
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.plot_surface(x1, x2, f, rstride=1, cstride=1, alpha=0.5)
print(fcf)
print(scf)
print(label)
ax.scatter(fcf, scf, label, c="black")
ax.set_xlabel("FC_F")
ax.set_ylabel("SC_F")
ax.set_zlabel(r"$\Delta$ NSA")
# plt.show()
ax.text(0.1,0.5,0.4, r"$\Delta$ NSA = 0.46997 * FC_F +  0.20146 * SC_F - 0.18815")
plt.savefig("3d.png")