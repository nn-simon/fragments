import cv2
import numpy as np
import sys
import os
import string

fh = open("voc.txt")
cnt = 0
num_classes = 10
lines = fh.readlines()
length = len(lines)
images = np.zeros((length, 224, 224, 3), dtype=np.float32)
labels = np.zeros((length, num_classes), dtype=np.float32)
for line in lines:
    pos = line.find(' ')
    tp = string.atoi(line[pos+1:-1])
    labels[cnt, tp] = 1
    img = cv2.imread(line[0:pos])
    images[cnt, :, :, :] = cv2.resize(img, (224, 224))
    cnt = cnt + 1
    print cnt
   
np.savez('voc2012', images = images, labels = labels) 
