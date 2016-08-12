import tensorflow as tf
import numpy as np
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

def permute(data, W_sp, length, times):
    flat = tf.reshape(data, [-1, length])
    trans = tf.transpose(flat)
    permute = tf.sparse_tensor_dense_matmul(W_sp, trans)
    return tf.reshape(tf.transpose(permute), [-1, length * times, 1, 1])

sess = tf.InteractiveSession()
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])
y_ = tf.placeholder(tf.float32, shape=[None, 159])

x_image = tf.reshape(x, [-1,28,28,1])

W_conv1 = weight_variable([5, 5, 1, 47])
b_conv1 = bias_variable([47])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

length1 = 14 * 14 * 47
W_sp = W_sparse(length1, 1);
permute1 = permute(h_pool1, W_sp, length1, 1)

W_conv2 = weight_variable([23, 1, 1, 7])
b_conv2 = bias_variable([7])
tp_conv2 = tf.nn.conv2d(permute1, W_conv2, strides=[1, 5, 1, 1], padding='SAME') + b_conv2
h_conv2 = tf.nn.relu(tp_conv2)
h_pool2 = tf.nn.avg_pool(h_conv2, [1, 17, 1, 1], strides=[1, 5, 1, 1], padding='SAME')

#batch = mnist.train.next_batch(50)
#sess.run(tf.initialize_all_variables())
#print h_pool2.eval(feed_dict={x: batch[0]}).shape

length2 = 369 * 7
W_sp2 = W_sparse(length2, 1)
permute2 = permute(h_pool2, W_sp2, length2, 1)

W_conv3 = weight_variable([17, 1, 1, 3])
b_conv3 = bias_variable([3])
tp_conv3 = tf.nn.conv2d(permute2, W_conv3, strides=[1, 7, 1, 1], padding='SAME') + b_conv3
#h_conv3 = tf.nn.tanh(tp_conv3)
h_pool3 = tf.nn.avg_pool(tp_conv3, [1, 17, 1, 1], strides=[1, 7, 1, 1], padding='SAME')

#batch = mnist.train.next_batch(50)
#sess.run(tf.initialize_all_variables())
#print h_pool3.eval(feed_dict={x: batch[0]}).shape

length3 = 159
y_conv=tf.nn.softmax(tf.reshape(h_pool3, [-1, length3]))
tp = range(0, 10) * 16;
maps = np.random.permutation(tp[0:length3]).tolist()

def adjust(y, nclasses, y_expand, num_expand):
    stat = np.zeros((nclasses, num_expand))
    ncases = y.size
    maps = np.zeros(num_expand)
    for nc in range(0, nclasses):
        tp = y_expand[y == nc]
        for ne in range(0, num_expand):
            stat[nc, ne] = sum(tp == ne)

    for ne in range(0, num_expand):
        tp = np.argmax(stat[ne % nclasses, :])
        maps[tp] = ne % nclasses
        stat[:, tp] = -1

    return maps.astype(int).tolist()

def y2y_(y, maps):
    dim = len(maps)
    ncases = y.size
    y_expand = np.zeros((ncases, dim))
    for i in range(0, ncases):
        pos = (maps == y[i])
        y_expand[i, pos] = 1.0 / sum(pos)
    return y_expand

def accuracy_expand(y, y_expand, maps):
    ncases = y.size
    cnt = 0
    for i in range(0, ncases):
        if y[i] == maps[y_expand[i]]:
            cnt = cnt + 1
    return cnt * 1.0 / ncases

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
y_pred = tf.argmax(y_conv, 1)
y_max = tf.argmax(y, 1)
sess.run(tf.initialize_all_variables())
for i in range(200000):
  batch = mnist.train.next_batch(50)
  yy = y_max.eval(feed_dict={y: batch[1]})
  if i%100 == 0:
    yy_pred = y_pred.eval(feed_dict={x: batch[0]}) 
    train_accuracy = accuracy_expand(yy, yy_pred, maps);
    #train_accuracy = accuracy.eval(feed_dict={
    #    x:batch[0], y_: batch[1]})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  yy_ = y2y_(yy, maps)
  train_step.run(feed_dict={x: batch[0], y_: yy_})

test_yy = y_max.eval(feed_dict={y: mnist.test.labels})
test_pred = y_pred.eval(feed_dict={x: mnist.test.images})
test_accuracy = accuracy_expand(test_yy, test_pred, maps)
print("test accuracy %g"%test_accuracy)
