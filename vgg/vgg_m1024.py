from kaffe.tensorflow import Network

class VGG_CNN_M_1024(Network):
    def setup(self):
        (self.feed('data')
             .conv(7, 7, 96, 2, 2, padding='VALID', name='conv1')
             .lrn(2, 0.0001, 0.75, name='norm1')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
             .conv(5, 5, 256, 2, 2, padding='SAME', name='conv2')
             .lrn(2, 0.0001, 0.75, name='norm2')
             .max_pool(3, 3, 2, 2, name='pool2')
             .conv(3, 3, 512, 1, 1, name='conv3')
             .conv(3, 3, 512, 1, 1, name='conv4')
             .conv(3, 3, 512, 1, 1, name='conv5')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
             .fc(4096, name='fc6')
             .fc(1024, name='fc7')
             .fc(1000, relu=False, name='fc8')
             .softmax(name='prob'))
