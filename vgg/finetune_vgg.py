# Import the converted model's class
# import finetune_vgg as fv
import numpy as np
import tensorflow as tf

from vgg_net import VGG_FACE_16_layer as VGG_FACE

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
net = VGG_FACE({'data': images})

fc7 = net.layers['fc7']
W_softmax = weight_variable([4096, num_classes])
b_softmax = bias_variable([num_classes])

y_ = tf.nn.softmax(tf.matmul(fc7, W_softmax) + b_softmax)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_, labels), 0)
opt = tf.train.RMSPropOptimizer(0.001)
train_op = opt.minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
print 'loading weights ...'
net.load('vgg_net.npy', sess)

print 'loading data ...'
data = np.load('data.npz')
in_images = data['images']
in_labels = data['labels']
data.close()
for i in range(1000):
    print i,
    train_op.run(feed_dict={images: in_images, labels: in_labels})
