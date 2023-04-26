import pandas as pd
import numpy as np
import os
father_path = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "csv_output")
# father_path = os.path.join(father_path, "connectome_fmri_o_443")

output = []
for f,_,files in os.walk(father_path):
    for file in files:
        path = os.path.join(f, file)
        s1 = pd.read_csv(path)
        s2 = s1.to_numpy()[:, 1:]
        s2 = np.reshape(s2, [-1, 18 * 18])
        output.append(s2)

output = np.concatenate(output, axis=0)
output = pd.DataFrame(output)
output.to_csv("final_result")




