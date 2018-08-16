from STResNet3D import STResNet3D
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnistpath = 'E:/Tim/Download/Chrome/handwrite'

fillspace = [x for x in ' .:-o?O%$@']
print(fillspace)
mnist = input_data.read_data_sets(mnistpath, one_hot=True)

def onehot2num(label):
    for i in range(len(label)):
        if label[i] != 0:
            return i
    return -1

def printarray(array):
    for i in range(len(array)):
        for j in range(len(array[i])):
            n = int(array[i][j] * 10)
            if n > 0:
                n = n % 10
                print(fillspace[n] + ' ', end='')
            elif n < 0:
                print('` ', end='')
            else:
                print('  ', end='')
        print('  |  ', end='')
        for j in range(len(array[i])):
            n = int(array[i][j] * 10)
            if n > 0:
                n = n % 10
                print(str(n) + ' ', end='')
            elif n < 0:
                print('` ', end='')
            else:
                print('  ', end='')
        print('')
    
def createbatch(x, y):
    classify = [[] for i in range(10)]
    l = len(x)
    for i in range(l):
        index = onehot2num(y[i])
        classify[index].append(x[i])
    for i in range(10):
        classify[i] = np.array(classify[i][:4320])
    return classify
    
if __name__ == '__main__':

    # test over fitting
    x, y = mnist.train.next_batch(60000)
    l = 10
    LR = 1e-5
    cimg = createbatch(x, y)
    #cimg = cimg.reshape((28, 28, 72, 1, -1))
    cimg = [img.reshape((-1, 72, 3, 28, 28)) for img in cimg]
    cimg = [np.swapaxes(img, 2, 3) for img in cimg]
    cimg = [np.swapaxes(img, 3, 4) for img in cimg]
    
    x_c = tf.placeholder(tf.float32, [None, 72, 28, 28, 3])
    x_p = tf.placeholder(tf.float32, [None, 72, 28, 28, 3])
    x_t = tf.placeholder(tf.float32, [None, 72, 28, 28, 3])
    
    y = tf.placeholder(tf.float32, [None, 72, 28, 28, 1])
    #y = tf.placeholder(tf.float32, [None, 28 * 28])
    
    resnet = STResNet3D(x_c, x_p, x_t, name='test')
    
    
    #a = tf.reshape(resnet, [-1, 72*28*28])
    #b = tf.layers.dense(a, 28*28)
    loss = tf.losses.mean_squared_error(y, resnet)
    #loss = tf.losses.mean_squared_error(y, b)
    
    train_op = tf.train.AdamOptimizer(LR).minimize(loss)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print('STResNet shape:', resnet.shape)
    print('cimg shape:', cimg[0][0:1].shape)
    print('cimg shape:', cimg[3][0:1, ::, ::, ::, 0:1].shape)
    printarray(cimg[3][0, 0, ::, ::, 0].reshape((28, 28)))
    for i in range(10000):
        _, loss_ = sess.run([train_op, loss], {x_c: cimg[0][0:1],
                                               x_p: cimg[1][0:1],
                                               x_t: cimg[2][0:1],
                                               y: cimg[3][0:1, ::, ::, ::, 0:1]})
                                               #y: [[1.5]]})
        print('.', end='')
        if i % 50 == 0:
            print('\nstep:', i, 'loss', loss_)
            
            img = sess.run(resnet, {x_c: cimg[0][0:1],
                                    x_p: cimg[1][0:1],
                                    x_t: cimg[2][0:1],})
            '''
            print(img = sess.run(b, {x_c: cimg[0][0:1],
                               x_p: cimg[1][0:1],
                               x_t: cimg[2][0:1],}))
            
            '''
            img = np.array(img)[0:1, 0, ::, ::, 0:1].reshape((28, 28))
            printarray(img)
            
    
    #
    
    '''
    for img in cimg:
        print(img.shape)
        printarray(img[0, 0, ::, ::, 0])
        printarray(img[3, 2, ::, ::, 1])
    '''    
        
    '''    
    printarray(cimg[::, ::, 0, 0, 3])
    printarray(cimg[::, ::, 3, 0, 5])
    '''
    '''
    for i in range(len(cimg)):
        print(i, len(cimg[i]))
        for j in range(2):
            image = cimg[i][j].reshape(28, 28)
            printarray(image)
    '''
    
    '''
    for i in range(l):
        print("====================================")
        xx = np.array(x).reshape((-1, 28, 28))
        #print(xx[i])
        printarray(xx[i])
        print(onehot2num(y[i]))
    '''


    '''
    # test the res net
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
    '''
    # test the unit
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