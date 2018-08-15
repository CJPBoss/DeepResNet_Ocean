from STResNet3D import _bn_relu_conv3d
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    matx = np.random.normal(size=1000).reshape((-1, 10, 10, 10, 1))
    matx = tf.Variable(matx)
    print(matx)
    print('============================')
    print('relu conv')
    a = _bn_relu_conv3d(matx, filters=10, kernel_size=3, strides=1, bn=False, name='Test')
    merged = tf.summary.merge_all()
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run(matx)[-1][-1][-1])
    print(sess.run(a)[-1][-1][-1])
    
    writer = tf.summary.FileWriter('test/residual_unit', sess.graph)
    '''
    sess = tf.Session()
    print('============================')
    #matx = np.random.normal(size=1000).reshape((-1, 10, 10, 10, 1))
    matx = np.arange(27).reshape((-1, 3, 3, 3, 1))
    matx = tf.Variable(matx, dtype=tf.float32)
    #sess.run(tf.initialize_all_variables())
    c3 = tf.layers.conv3d(
        inputs=matx,
        filters=1,
        kernel_size=3,
        strides=1,
        activation=None,
        padding='same'
    )
    sess.run(tf.global_variables_initializer())
    print(sess.run(matx)[-1][-1][-1])
    print(sess.run(c3)[-1][-1][-1])
    '''