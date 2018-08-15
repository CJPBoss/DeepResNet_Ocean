from STResNet3D import _bn_relu_conv3d
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    matx = np.random.normal(size=1000).reshape((-1, 10, 10, 10, 1))
    matx = tf.constant(matx)
    ##print(matx)
    sess = tf.Session()
    print('============================')
    #print(sess.run(matx))
    #print('relu conv')
    a = _bn_relu_conv3d(matx, filters=10, kernel_size=(3, 3, 3), strides=(1, 1, 1), bn=False, name='Test')
    print(sess.run(a))