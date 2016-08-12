import tensorflow as tf
import numpy as np

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

def rand_fc(data, length, conv_para, pool_para)
    W_sp = W_sparse(length, 1);
    p = permute(data, W_sp, length, 1)

    W = weight_variable([conv_para[0], 1, 1, conv_para[1]])
    b = bias_variable(conv_para[1])
    mid = tf.nn.conv2d(p, W, strides=[1, in_para[2], 1, 1], padding='SAME') + b
    conv = tf.nn.relu(mid)
    pool = tf.nn.max_pool(conv, [1, pool_para[0], 1, 1], strides=[1, pool_para[1], 1, 1], padding='SAME')

    return pool

from vgg_ilsvrc import VGG_ILSVRC_16_layer as VGG16

sess = tf.InteractiveSession()

num_classes = 1000
reduce_length = 5000
x = tf.placeholder(tf.float32, [None, 224, 224, 3])
y_ = tf.placeholder(tf.float32, [None, num_classes])

print 'loading structure ...'
net = VGG16({'data': images})

W_conv1 = weight_variable([3, 3, 512, reduce_length])
b_conv1 = bias_variable([reduce_length])
val_conv1 = tf.nn.relu(conv2d(net.layers['conv5_2'], W_conv1) + b_conv1)
val = tf.reduce_max(val_conv1, reduction_indices = [1,2])

length1 = reduce_length
W_sp = W_sparse(length1, 1);
permute1 = permute(val, W_sp, length1, 1)

W_conv2 = weight_variable([157, 1, 1, 23])
b_conv2 = bias_variable([23])
tp_conv2 = tf.nn.conv2d(permute1, W_conv2, strides=[1, 29, 1, 1], padding='SAME') + b_conv2
h_conv2 = tf.nn.relu(tp_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, [1, 17, 1, 1], strides=[1, 5, 1, 1], padding='SAME')

#batch = mnist.train.next_batch(50)
#sess.run(tf.initialize_all_variables())
#print h_pool2.eval(feed_dict={x: batch[0]}).shape

length2 = ...
W_sp2 = W_sparse(length2, 1)
permute2 = permute(h_pool2, W_sp2, length2, 1)

W_conv3 = weight_variable([203, 1, 1, 23])
b_conv3 = bias_variable([23])
tp_conv3 = tf.nn.conv2d(permute2, W_conv3, strides=[1, 29, 1, 1], padding='SAME') + b_conv4
h_conv3 = tf.nn.relu(tp_conv3)
h_pool3 = tf.nn.max_pool(h_conv3, [1, 17, 1, 1], strides=[1, 5, 1, 1], padding='SAME')

length3 = ...
W_sp3 = W_sparse(length3, 1)
permute3 = permute(h_pool3, W_sp3, length3, 1)

W_conv4 = weight_variable([113, 1, 1, 1000])
b_conv4 = bias_variable([1000])
tp_conv4 = tf.nn.conv2d(permute3, W_conv4, strides=[1, 17, 1, 1], padding='SAME') + b_conv4
#h_conv3 = tf.nn.tanh(tp_conv3)
#h_pool3 = tf.nn.avg_pool(tp_conv3, [1, 17, 1, 1], strides=[1, 7, 1, 1], padding='SAME')

#batch = mnist.train.next_batch(50)
#sess.run(tf.initialize_all_variables())
#print h_pool3.eval(feed_dict={x: batch[0]}).shape

y_conv=tf.nn.softmax(tf.reduce_max(tp_conv4, reduction_indices = [1, 2]))
#y_conv=tf.nn.softmax(tf.reshape(h_pool3, [-1, length3]))

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

print 'loading weights ...'
net.load('vgg_ilsvrc.npy', sess)

print 'loading data ...'
data = np.load('voc2012.npz')
in_images = data['images']
data.close()

maxepoch = 100
batchsize = 100
maxbatch = in_images.shape[0] / batchsize

saver = tf.train.Saver()

for epoch in xrange(maxepoch):
  order = np.random.permutation(in_images.shape[0])
  for batch in xrange(maxbatch):
      s = batch * batchsize
      e = (batch + 1)*batchsize
      imgs = in_images[s:e, :, :, :]
      labels = in_labels[s:e, :]
      train_step.run(feed_dict={x: imgs, y_: labels})
      if batch%100 == 0:
          train_accuracy = accuracy.eval(feed_dict={
              x: imgs, y_: labels})
          print("epoch %d step %d, training accuracy %g"%(epoch, batch, train_accuracy))
  train_accuracy = accuracy.eval(feed_dict={
      x: imgs, y_: labels})
  print("epoch %d, training accuracy %g"%(epoch, train_accuracy))
  save_path = saver.save(sess, 'imagenet_vgg' + str(epoch))
  print("Model saved in file: %s" % save_path)
