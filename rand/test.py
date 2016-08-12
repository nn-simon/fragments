import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

sess = tf.InteractiveSession()
batchsize = 50
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [batchsize,28,28,1])
vec_flat = tf.reshape(x_image, [batchsize, -1])
shape = tf.shape(vec_flat)
sess.run(tf.initialize_all_variables())
print shape.eval()

