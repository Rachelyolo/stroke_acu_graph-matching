import numpy as np

# 1：fmri_precision_health_0.py
#   输入：connect_fmri_443 分为health-post & health-pre
#        health-post 缺第1和第17个人 所以总共的人数是

import fmri_precision_health_0 as fh0
import fmri_precision_stroke_0 as fs0
import fmri_costmap_health_1 as fh1
import fmri_costmap_stroke_1 as fs1

import fmri_heatmap_health_2_precision as fh2
import fmri_heatmap_stroke_2_precision as fs2

import fmri_heatmap_health_2_single as fh3
import fmri_heatmap_stroke_2_single as fs3

import ratio_research as ra
# for gamma in np.linspace(0.5, 0.6, 20):
#     print(gamma)
gamma = "fmri_v3"
print("文件名的后缀为：", gamma)
print("正在跑fmri_precision_health_0.py")
path1, path2 = fh0.run(gamma)
print("fmri_precision_health_0.py finish")
print("--------------------------------------------------------------------------------------------------------------------------------------------------")
print("正在跑fmri_costmap_health_1.py")
path_h = fh1.run(gamma, path1, path2)
print("fmri_costmap_health_1.py finish")
print("--------------------------------------------------------------------------------------------------------------------------------------------------")
print("正在跑fmri_precision_stroke_0.py")
path1, path2 = fs0.run(gamma)
print("fmri_precision_stroke_0 finish")
print("--------------------------------------------------------------------------------------------------------------------------------------------------")
print("正在跑fmri_costmap_stroke_1.py")
path_s = fs1.run(gamma, path1, path2)
print("fmri_costmap_stroke_1 finish")
print("--------------------------------------------------------------------------------------------------------------------------------------------------")
print("正在跑fmri_heatmap_health_2_precision.py")
path_node, path_network = fh2.run(gamma, path_h)
print("fmri_heatmap_health_2_precision.py finish")
print("--------------------------------------------------------------------------------------------------------------------------------------------------")

print("正在跑fmri_heatmap_stroke_2_precision.py")
paths_node, paths_network = fs2.run(gamma, path_s)
print("fmri_heatmap_stroke_2_precision.py finish")
print("--------------------------------------------------------------------------------------------------------------------------------------------------")
print("正在跑ratio_research.py")
flag_path = ra.run(gamma, path_node, paths_node)

print("ratio_research.py finish")
print("--------------------------------------------------------------------------------------------------------------------------------------------------")
# fh3.run()
# print("fh3 finish")
print("正在跑fmri_heatmap_stroke_2_single.py")
fs3.run(gamma, path_s, flag_path)
print("fmri_heatmap_stroke_2_single.py finish")




























