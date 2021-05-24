from STResNet3D import STResNet3D
import tensorflow as tf
import numpy as np

import sys
import os
import time

#MAINDIRPATH = os.path.abspath(os.path.join(os.getcwd(), ".."))+'/'#'E:/Tim/onedrive/Code/py_desktop/DeepResNet_Ocean/'
MAINDIRPATH = '/home/lzt/Deep_Ocean'
sys.path.append(MAINDIRPATH)
from datasets.load_temperature import create_batch, load_data
from datasets.ReadBOA_Argo_mat import SaveImage

len_proc = 3
imgdepth = 61
imgheight = 20
imgwidth = 20
channels = (3, 2, 2)
intervals = (1, 50, 250)
kernel_size = ((3, 3, 3), (3, 3, 3), (3, 3, 3))
filters = (32, 32, 32)
strides=((1, 1, 1), (1, 1, 1), (1, 1, 1))
num_res_units = (3, 2, 2)
batch_size = 50
learn_rate = 1e-6
train_loop = 5000
outstep = 100
stoplimit = 5
trainsize = 9.0/10
isTrained = True
version = 'v1.03'
alldata = None
savename = 'temperature/saver/'
savePath = savename
sessname = 'temperature/board/'
sessPath = sessname
if not os.path.exists(savePath):
    os.makedirs(savePath)

def getcorrectsavepath(dirpath):
    with open(dirpath + 'checkpoint', 'r') as checkpoint:
        info = checkpoint.readline()
        file = info.split('"')
        print(file, file[1])
        return dirpath + file[1]
    
if __name__ == '__main__':
    alldata = load_data()
    datasize = len(alldata)
    splitindex = datasize - int(datasize * trainsize)
    traindata = alldata[splitindex:]
    testdata = alldata[:splitindex]
    batches = create_batch(batch_size, channels, intervals, alldata)
    
    x_input = []
    for i in range(len_proc):
        x_ph = tf.placeholder(tf.float32, [None, imgdepth, imgheight, imgwidth, channels[i]])
        x_input.append(x_ph)
    y = tf.placeholder(tf.float32, [None, imgdepth, imgheight, imgwidth, 1])
    
    resnet = STResNet3D(x_input, kernel_size, filters, strides, num_res_units, 'temperature')
    
    loss = tf.losses.mean_squared_error(y, resnet)
    tf.summary.scalar('loss', loss)
    
    train_op = tf.train.AdamOptimizer(learn_rate).minimize(loss)
    merged = tf.summary.merge_all()
    
    saver = tf.train.Saver(max_to_keep=3)
    
    sess = tf.Session()
    writer = tf.summary.FileWriter(sessPath, sess.graph)
    
    
    if isTrained:
        try:
            sp = getcorrectsavepath(savePath)
            print('[+] load model:', sp)
            saver.restore(sess, sp)
        except Exception:
            print('[-] Saved model not found:', savePath)
            print("[+] init model")
            sess.run(tf.global_variables_initializer())
    else:
        print("[+] init model")
        sess.run(tf.global_variables_initializer())
    
    starttime = time.time()
    min_loss = 100
    lesstime = 0
    
    for i in range(train_loop):
        one_batch = create_batch(batch_size, channels, intervals, traindata)
        outdict = {}
        for j in range(len(x_input)):
            outdict[x_input[j]] = one_batch[j]
        outdict[y] = one_batch[-1]
        
        _, loss_ = sess.run([train_op, loss], outdict)
        if (i+1)%outstep == 0 or i == 0:
            print('\n==========================\n[+] ', end='')
            if i != 0:
                currenttime = time.time()
                usedtime = currenttime - starttime
                needtime = usedtime * (train_loop - i) / i
                print('used time: [%d:%02d]'%(usedtime / 60, usedtime % 60), end=' ')
                print('ext time: [%d:%02d]'%(needtime / 60, needtime % 60), end=' ')
                
            one_batch = create_batch(batch_size, channels, intervals, testdata)
            indict = {}
            for j in range(len(x_input)):
                indict[x_input[j]] = one_batch[j]
            indict[y] = one_batch[-1]
            
            res, loss__ = sess.run([resnet, loss], indict)
            print('step:[', i, ']loss:[', loss__, ']\n+++++++++++\n\n')
            imgpath = 'temperature/image' + version + '/step_' + str(i) + '/'
            if not os.path.exists(imgpath):
                os.makedirs(imgpath)
            SaveImage(np.array(res[0]).reshape((61, 20, 20)) * 30., imgpath + 'p')
            SaveImage(np.array(one_batch[-1][0]).reshape((61, 20, 20)) * 30., imgpath + 'y')
            if loss__ < min_loss:
                min_loss = loss__
                save_path = saver.save(sess, savePath, global_step=i + 1)
                #saver.restore(sess, savePath)
                #print(save_path)
                lesstime = 0
            else :
                lesstime += 1
                if lesstime >= stoplimit:
                    print('[+] early stop, less loss :', min_loss)
                    break
