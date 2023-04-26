import numpy as np

# 1：fmri_precision_health_0.py
#   输入：connect_fmri_443 分为health-post & health-pre
#        health-post 缺第1和第17个人 所以总共的人数是




import dti_flip_health_0 as fh0
import dti_flip_stoke_0 as fs0
import dti_costmap_health_1 as fh1
import dti_costmap_stroke_1 as fs1

import dti_heatmap_health_2 as fh2
import dti_heatmap_stroke_2 as fs2

import ratio_research_dti as ra

# import fmri_heatmap_health_2_single as fh3
import dti_heatmap_stroke_single as fs3
# for gamma in np.linspace(0.5, 0.6, 20):
#     print(gamma)
gamma = "dti_v5_health_xiaohuazhao11"
print("文件名的后缀为：", gamma)
print("正在跑dti_flip_health_0.py")
path1, path2 = fh0.run(gamma)
print("dti_flip_health_0.py finish")
print("--------------------------------------------------------------------------------------------------------------------------------------------------")
print("正在跑dti_costmap_health_1.py")
path_h = fh1.run(gamma, path1, path2)
print("dti_costmap_health_1.py finish")
print("--------------------------------------------------------------------------------------------------------------------------------------------------")
print("正在跑dti_flip_stoke_0.py")
path1, path2 = fs0.run(gamma)
print("dti_flip_stoke_0 finish")
print("--------------------------------------------------------------------------------------------------------------------------------------------------")
print("正在跑dti_costmap_stroke_1.py")
path_s = fs1.run(gamma, path1, path2)
print("dti_costmap_stroke_1 finish")
print("--------------------------------------------------------------------------------------------------------------------------------------------------")
print("正在跑dti_heatmap_health_2.py")
path_node = fh2.run(gamma, path_h)
print("dti_heatmap_health_2.py finish")
print("--------------------------------------------------------------------------------------------------------------------------------------------------")

print("正在跑dti_heatmap_stroke_2.py")
paths_node = fs2.run(gamma, path_s)
print("dti_heatmap_stroke_2.py finish")
print("--------------------------------------------------------------------------------------------------------------------------------------------------")
print("正在跑ratio_research_dti.py")
flag_path = ra.run(gamma, path_node, paths_node)

print("ratio_research_dti.py finish")
print("--------------------------------------------------------------------------------------------------------------------------------------------------")
fs3.run(gamma, path_s, flag_path)
print("fs3 finish")
# print("正在跑fmri_heatmap_stroke_2_single.py")
# fs3.run(gamma, path_s, flag_path)
# print("fmri_heatmap_stroke_2_single.py finish")




























