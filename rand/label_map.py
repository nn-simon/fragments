import tensorflow as tf
import numpy as np

def mapl():
    return

def adjust(y, num, pred, expand_num):
    for idx in range(0, expand_num):
        pos = (pred == idx)
        tp_max = 0
        tp_label = 0
        for label in range(0, num):
            tp = np.sum(y[pos])
            if (tp > tp_max):
                tp_max = tp
                tp_label = label
        maps[idx] = tp_label
 
