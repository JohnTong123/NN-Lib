
'''
Verifying images in the dataset, feel free to ignore
'''

import numpy as np
from PIL import Image
f = open(r"C:\Users\johna\cnn\finished\data_set\train\train.txt","r")
gg = f.readline()
gg = gg.split(" ")

gg = [int(i) for i in gg[1:-1]]
print(len(gg))
gg = np.array(gg)
gg = gg.reshape((28,28)).astype(np.uint8)
# print(gg)
print(gg.shape)
# for i in gg:
    # print(i)
# gg = gg/255
Image.fromarray(gg).save('gg.png')
# cv2.imshow(gg)
