from petrel_well_file_readers import *
import re
import pandas as pd
folder=r"D:\SoftwareWebApps\Python\geophysics\inversion&spect_decomp\d11_data\\"
# dev_file=folder+'nec25_a1_dev.dat'
# print(read_dev(dev_file))
chkt_file=folder+"nec25_a1_chkt.dat"
print(read_chkt(chkt_file))