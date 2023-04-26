# dti

# fmri
import numpy as np
from skbio.stats.distance import mantel
s1 = np.load("../threold_research/fmri_health_node.npy")
s2 = np.load("../threold_research/fmri_stroke_node.npy")

s3 = np.load("../threold_research/dti_health_node.npy")
s4 = np.load("../threold_research/dti_stroke_node.npy")


print(s2)
print(s1)
threold  = 5
for i in range(s1.shape[0]):
    for j in range(s2.shape[1]):
        if i == j:
            continue
        if s1[i, j] >=5:
            s2[i, j] = 0

print(np.count_nonzero(s2) / 443 / 443)






# def foo(a, b):
#     n, _ = b.shape
#     ans = b
#     diff = np.abs(a - b).sum()
#     for i in range(n):
#         tmp = np.roll(b, i, (0, 1))
#         print(tmp.shape)
#         input()
#         tmp_diff = np.abs(a - tmp).sum()
#         if tmp_diff < diff:
#             diff = tmp_diff
#             ans = tmp
#     return ans, diff
# _, diff = foo(s1, s3)

# print(diff)
