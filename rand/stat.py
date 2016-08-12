import numpy as np

def stat(y, numclasses, y_expand, num_expand):
    ss = np.zeros((numclasses, num_expand))
    for nc in range(0, numclasses):
        tp = y_expand[(y == nc)]
        for ne in range(0, num_expand):
            ss[nc, ne] = np.sum(tp == ne)

    for ne in range(0, num_expand):
        print '%03d'%ne,
        for nc in range(0, numclasses):
            print '% 5d'%ss[nc, ne],
        print ''

