# Import the converted model's class
# import finetune_vgg as fv
import numpy as np
import tensorflow as tf

from vgg_m1024 import VGG_CNN_M_1024 as VGG_M1024

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

num_classes = 10
images = tf.placeholder(tf.float32, [None, 224, 224, 3])
labels = tf.placeholder(tf.float32, [None, num_classes])
print 'loading structure ...'
net = VGG_M1024({'data': images})

pool5 = net.layers['pool5']

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
print 'loading weights ...'
net.load('vgg_m1024.npy', sess)

print 'loading data ...'
data = np.load('voc2012.npz')
in_images = data['images']
data.close()
pool5_val = np.zeros((17125, 6, 6, 512), dtype=np.float32)

for i in range(0, 17):
    s = i * 1000
    e = (i + 1) * 1000
    pool5_val[s:e, :, :, :] = pool5.eval(feed_dict={images: in_images[s:e, :, :, :]})

pool5_val[e:17125, :, :, :] = pool5.eval(feed_dict={images: in_images[e:17125, :, :, :]})
np.save('pool5', pool5_val)
