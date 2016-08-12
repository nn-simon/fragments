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

def rand_fc(data, length, conv_para, pool_para):
    W_sp = W_sparse(length, 1);
    p = permute(data, W_sp, length, 1)

    W = weight_variable([conv_para[0], 1, 1, conv_para[1]])
    b = bias_variable([conv_para[1]])
    mid = tf.nn.conv2d(p, W, strides=[1, conv_para[2], 1, 1], padding='SAME') + b
    conv = tf.nn.relu(mid)
    pool = tf.nn.max_pool(conv, [1, pool_para[0], 1, 1], strides=[1, pool_para[1], 1, 1], padding='SAME')

    return pool

print 'loading data ...'
data = np.load('cifar_train.npz')
in_images = data['images']
in_labels = data['labels']
data.close()

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W_conv1 = weight_variable([5, 5, 3, 16])
b_conv1 = bias_variable([16])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([3, 3, 16, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#img = in_images[0:50, :, :, :]
#sess.run(tf.initialize_all_variables())
#print layer4.eval(feed_dict={x: img}).shape

length1 = 8 * 8 * 64
layer2 = rand_fc(h_pool2, length1, [19, 23, 11], [13, 2]);

length2 = 187 * 23 
layer3 = rand_fc(layer2, length2, [17, 19, 11], [11, 5]);

length3 = 79 * 19
layer4 = rand_fc(layer3, length3, [23, 11, 13], [11, 5]);

#img = in_images[0:50, :, :, :]
#sess.run(tf.initialize_all_variables())
#print layer4.eval(feed_dict={x: img}).shape

length4 = 24 * 11 
layer4_flat = tf.reshape(layer4, [-1, length4])
W_fc2 = weight_variable([length4, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(layer4_flat, W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

maxepoch = 1000
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
  if epoch%10 == 0:
      save_path = saver.save(sess, 'cifar_fc_2' + str(epoch))
      print("Model saved in file: %s" % save_path)
