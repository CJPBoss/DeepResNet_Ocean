import numpy as np
import tensorflow as tf
from STResNet3D import STResNet3D
import sys
path = 'E:/Tim/onedrive/Code/py_desktop/DeepResNet_Ocean/'
sys.path.append(path + 'datasets') # path to dir
from loadtemp import DataSet
from ReadBOA_Argo_mat import SaveImage
import os
import time


LR = 1e-6
batch_size = 50
index = None
trainloop = 5000
outstep = 50
isTrained = True
version = 'v1.01'
savePath = "test/test_" + version + "/save_net.ckpt"

if __name__ == '__main__':
    #dataset = DataSet(path + 'datasets/monthly_data_area.mat')
    dataset = DataSet(r'E:\Tim\onedrive\Code\py_desktop\DeepResNet_Ocean\datasets\monthly_data_area.mat')
    dataset.init_cs(channels=(2, 2, 2), steps=(1, 3, 12))
    
    x_c = tf.placeholder(tf.float32, [None, 26, 20, 20, 2])
    
    x_p = tf.placeholder(tf.float32, [None, 26, 20, 20, 2])
    x_t = tf.placeholder(tf.float32, [None, 26, 20, 20, 2])
    
    y = tf.placeholder(tf.float32, [None, 26, 20, 20, 1])
    
    resnet = STResNet3D(x_c, x_p, x_t,
                        filters=(32 ,32, 32),
                        num_res_units=(2, 1, 1),
                        name='test')
    
    loss = tf.losses.mean_squared_error(y, resnet)
    
    tf.summary.scalar('loss', loss)
    
    train_op = tf.train.AdamOptimizer(LR).minimize(loss)
    merged = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=3)
    
    sess = tf.Session()
    writer = tf.summary.FileWriter('test/temp', sess.graph)
    
    if isTrained:
        saver.restore(sess, savePath)
    else :   
        sess.run(tf.global_variables_initializer())
        
    print(resnet.shape)
    print(x_c.shape)
    factor = 1. / 30
    starttime = time.time()
    totalloop = trainloop / outstep
    min_loss = 10
    for i in range(trainloop):
        b_x, b_1, b_2, b_y, month = dataset.next_batch(batch_size, index)
        inputdict = {x_c: b_x, x_p: b_1, x_t: b_2}
        outputdict = {x_c: b_x, x_p: b_1, x_t: b_2, y: b_y}
        _, loss_ = sess.run([train_op, loss], outputdict)
        if (i+1)%outstep == 0 or i == 0:
            print("\n===================\n[+] ", end='')
            if i != 0:
                currenttime = time.time()
                usedtime = currenttime - starttime
                needtime = usedtime * (trainloop - i) / i
                print('used time: [%d:%02d]'%(usedtime / 60, usedtime % 60), end=' ')
                print('ext time: [%d:%02d]'%(needtime / 60, needtime % 60), end=' ')
            res = sess.run(resnet, inputdict)
            print('step:[', i, ']loss:[', loss_, ']\n+++++++++++\n\n')
            path = 'test/' + version + '/step_' + str(i) + '/' + month[0] + '/'
            if not os.path.exists(path):
                os.makedirs(path)
            SaveImage(np.array(res[0]).reshape((26, 20, 20)) * 30., path + 'p')
            SaveImage(np.array(b_y[0]).reshape((26, 20, 20)) * 30., path + 'y')
            if loss_ < min_loss:
                min_loss = loss_
                save_path = saver.save(sess, savePath, global_step=i + 1)
    
    
    
    