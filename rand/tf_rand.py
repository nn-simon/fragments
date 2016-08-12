import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def W_sparse(dim, times):
  import numpy as np
  tp = range(dim) * times
  idx = np.row_stack((range(dim * times), np.random.permutation(tp))).transpose().tolist()
  return tf.SparseTensor(indices=idx, values=[1.0] * (dim * times), shape=[dim * times, dim])

sess = tf.InteractiveSession()
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

x_image = tf.reshape(x, [-1,28,28,1])

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_pool2_trans = tf.transpose(h_pool2_flat)

W_sp = W_sparse(7*7*64, 3);
h_pool2_rand = tf.sparse_tensor_dense_matmul(W_sp, h_pool2_trans)
h_pool2_rand_trans = tf.reshape(tf.transpose(h_pool2_rand), [-1, 7*7*64*3, 1, 1])

W_conv3 = weight_variable([17, 1, 1, 1])
b_conv3 = bias_variable([1])
tp_conv3 = tf.nn.conv2d(h_pool2_rand_trans, W_conv3, strides=[1, 5, 1, 1], padding='SAME') + b_conv3
h_conv3 = tf.nn.relu(tp_conv3)
h_pool3 = tf.nn.avg_pool(h_conv3, [1, 11, 1, 1], strides=[1, 3, 1, 1], padding='SAME')
fc_in = tf.reshape(h_pool3, [-1, 628])

#xx = h_pool2_rand_trans.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
#tp1 = tp_conv3.eval(feed_dict={h_pool2_rand_trans: xx})
#hc3 = h_conv3.eval(feed_dict={tp_conv3: tp1})
#hp3 = h_pool3.eval(feed_dict={h_conv3: hc3})
#keep_prob = tf.placeholder(tf.float32)
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#

W_fc2 = weight_variable([628, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(fc_in, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1]})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

print("test accuracy %g"%accuracy.eval(feed_dict={
   x: mnist.test.images, y_: mnist.test.labels}))
