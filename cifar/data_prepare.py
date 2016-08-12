import cv2
import numpy as np
import sys
import os
import string

fh = open("cifar_train.txt")
cnt = 0
num_classes = 10
lines = fh.readlines()
length = len(lines)
images = np.zeros((length, 32, 32, 3), dtype=np.float32)
labels = np.zeros((length, num_classes), dtype=np.float32)
for line in lines:
    pos = line.find(' ')
    tp = string.atoi(line[pos+1:-1])
    labels[cnt, tp] = 1
    img = cv2.imread(line[0:pos])
    images[cnt, :, :, :] = img 
    cnt = cnt + 1
    print cnt
   
np.savez('cifar_train.npz', images = images, labels = labels) 
