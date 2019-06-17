import glob
import os
import shutil
import numpy as np

folder_name = '/home/yunhan/Documents/data/apollo/'

filenames = []
count = 0
for filename in glob.iglob('/home/yunhan/Documents/data/apollo/output_highway/images/**/*.jpg', recursive=True):
    print(count)
    shutil.move(filename, '/home/yunhan/Documents/data/detection/%05d.jpg' % count)
    count = count + 1
