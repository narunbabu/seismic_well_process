from petrel_well_file_readers import *
import numpy as np
import pandas as pd
# %matplotlib inline
from matplotlib import pyplot as plt
# from sklearn.linear_model import LinearRegression
import lasio
# from flex_log import FlexXY,FlexLog
# from seis_ampl_spectrum import *
folder=r'D:\SoftwareWebApps\Python\geophysics\inversion&spect_decomp\d11_data\\'
# folder=r"D:\Ameyem\python\inversion&spectraldecomp\d11_data\\"
# las_file=folder+'nec25_a1.las'
# las=lasio.read(las_file)

# well_tops_file=folder+'d11_welltops_payzones_220519_sai.dat'
# wt=read_welltops(well_tops_file)
# wt=wt.sort_values(['MD'])

well_dev_file=folder+'nec25_a1_dev.dat'
dev=read_dev(well_dev_file)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(12,24))
ax = fig.add_subplot(111, projection='3d')
ax.plot(dev['X'],dev['Y'],dev['Z'])
ax.plot(dev['X'],dev['Y'],-dev['MD'])
# plt.show()

# fig2 = plt.figure(figsize=(12,24))
# ax = fig.add_subplot(122, projection='3d')
# ax.plot(dev['X'],dev['Y'],dev['MD'])
# ax.set_zlim(max(dev['MD'].values), 0)
plt.show()