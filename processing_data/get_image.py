import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import pydicom
import re
from PIL import Image

patient_2d_path = './medical/'
out_real= './medical/'


DSA_array2 = pydicom.read_file(patient_2d_path)
array2 = cv2.normalize(DSA_array2.pixel_array.astype(float), None, 0.0, 1.0, cv2.NORM_MINMAX)
array2 = np.transpose(array2, (1,2,0))
DSA_img = array2
angle_list = DSA_array2.PositionerPrimaryAngleIncrement
angle2_list = DSA_array2.PositionerSecondaryAngleIncrement
#save
ii = range(133)
for index in ii:
    image_array = DSA_img[:,:,index]

    im = (image_array*255.0)
 
    #去掉上下黑边
    #im = im[116:908,:]

    cv2.imwrite(out_real+'/traindata_' +str(index) + '.png',im)

    with open(out_real + '/angle.txt','a') as f:
            f.write(str(index)+ ' '+str(angle_list[index])+ ' '+str(angle2_list[index])+'\n')

    print(index,angle_list[index])