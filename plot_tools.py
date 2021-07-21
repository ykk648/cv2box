import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


import os
n_list = []
for f in os.listdir('/workspace/151_cluster/nfs/dataset/deepfacelab/material/待处理/face512aligned'):
    f_p = os.path.join('/workspace/151_cluster/nfs/dataset/deepfacelab/material/待处理/face512aligned', f)
    for ff in os.listdir(f_p):
        f_p_p = os.path.join(f_p, ff)
        f_p_len = len(os.listdir(f_p_p))
        # print(f_p_len)
        n_list.append(f_p_len)


sns.set_style("darkgrid")
# x = np.random.normal(size=200)
sns.displot(n_list, color='y')
# plt.plot(np.arange(10))
plt.show()