from STResNet3D import STResNet3D
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    matxc = tf.Variable(np.random.normal(size=9000).reshape((-1, 10, 10, 10, 1)))
    matxp = tf.Variable(np.random.normal(size=9000).reshape((-1, 10, 10, 10, 1)))
    matxt = tf.Variable(np.random.normal(size=9000).reshape((-1, 10, 10, 10, 1)))
    #print(matx)
    print('============================')
    a = STResNet3D(
        matxc, matxp, matxt,
    )
    merged = tf.summary.merge_all()
    
    sess = tf.Session()
    print('============================')
    writer = tf.summary.FileWriter('test/residual_net', sess.graph)
    sess.run(tf.global_variables_initializer())
    #print(sess.run(matx)[-1][-1][-1])
    print(sess.run(a)[-1][-1][-1])
    
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