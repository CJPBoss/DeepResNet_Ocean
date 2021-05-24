from STResNet3D import STResNet3D
import tensorflow as tf
import numpy as np
from scipy.io import savemat

import sys
import os
import time

#MAINDIRPATH = os.path.abspath(os.path.join(os.getcwd(), ".."))+'/'#'E:/Tim/onedrive/Code/py_desktop/DeepResNet_Ocean/'
MAINDIRPATH = '/home/lzt/Deep_Ocean/'
sys.path.append(MAINDIRPATH)
from datasets.load_temperature import create_batch, load_data, get_lastest_batch
from datasets.ReadBOA_Argo_mat import SaveImage

len_proc = 3
imgdepth = 26
imgheight = 20
imgwidth = 20
channels = (7, 6, 6)
intervals = (1, 3, 6)
kernel_size = ((3, 3, 3), (3, 3, 3), (3, 3, 3))
filters = (32, 32, 32)
strides=((1, 1, 1), (1, 1, 1), (1, 1, 1))
num_res_units = (4, 4, 4)
batch_size = 64
learn_rate = 1e-6
train_loop = 20000
outstep = 100
stoplimit = 100
#trainsize = 10.0/10
isSingle = False
isTrained = True
isPredicting = False
lenPredict = 12
monthinter = 1
version = 'v1.25'
alldata = None

savename = 'result/temperature_%s/saver/'%(version)
savePath = savename
sessname = 'result/temperature_%s/board/'%(version)
sessPath = sessname

datapath = MAINDIRPATH + 'datasets/oridata.mat'#%(version)
depseq = np.array([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 200, 220, 240, 260, 280, 300])
depindex = [int(i) for i in depseq / 5]
print('datapath:'+datapath)


if not os.path.exists(savePath):
    os.makedirs(savePath)

def getcorrectsavepath(dirpath):
    with open(dirpath + 'checkpoint', 'r') as checkpoint:
        info = checkpoint.readline()
        file = info.split('"')
        print(file, file[1])
        return dirpath + file[1]
    
if __name__ == '__main__':
    print(datapath)
    alldata = load_data('data', datapath)
    datasize = len(alldata)
    splitindex = datasize - int(datasize * trainsize)
    '''
    traindata = alldata[splitindex:]
    testdata = alldata[:splitindex]
    '''
    traindata = alldata
    testdata = load_data('data', 'data_2017.mat')
    #batches = create_batch(batch_size, channels, intervals, alldata)
    
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
    
    if not isPredicting:
        if not isSingle:
            isPredicting = True
        print("Training...")
        starttime = time.time()
        min_loss = 100
        lesstime = 0
        for i in range(train_loop):
            #print(i, time.time())
            one_batch = create_batch(batch_size, channels, intervals, traindata)
            outdict = {}
            for j in range(len(x_input)):
                outdict[x_input[j]] = one_batch[j]
            outdict[y] = one_batch[-1]
            
            _, loss_ = sess.run([train_op, loss], outdict)
            if (i+1)%outstep == 0 or i == 0:
                print('\n==========================\n[+] step[ ', i, ' ]', end='')
                if i != 0:
                    currenttime = time.time()
                    usedtime = currenttime - starttime
                    needtime = usedtime * (train_loop - i) / i
                    print('used time: [%d:%02d]'%(usedtime / 60, usedtime % 60), end=' ')
                    print('ext time: [%d:%02d]'%(needtime / 60, needtime % 60))
                else:
                    print('used time: [%d:%02d]'%(usedtime / 60, usedtime % 60), end=' ')
                tedata = traindata
                print('')
                for m in range(12):
                #####################################################
                    indict = {}
                    one_batch = get_lastest_batch(channels, intervals, tedata)
                    for p in range(len(x_input)):
                        indict[x_input[p]] = one_batch[p]
                    indict[y] = tf.convert_to_tensor(testdata[-m + 1].reshape((1, imgdepth, imgheight, imgwidth, 1)))
                    res, _l = sess.run([resnet, loss], indict)
                    print('\t[\t' , i, '\t] loss: [\t', _l, '\t]')
                    tedata = np.vstack([testdata[-m+1].reshape((1, imgdepth, imgheight, imgwidth)), tedata])
                
                one_batch = create_batch(batch_size, channels, intervals, testdata)

                indict = {}
                for j in range(len(x_input)):
                    indict[x_input[j]] = one_batch[j]
                indict[y] = one_batch[-1]
                
                res, loss__ = sess.run([resnet, loss], indict)
                print('\tstep:[', i, ']loss:[', loss__, '], minimum loss[', min_loss, ']\n+++++++++++\n\n')
                
                '''
                with open('res.txt', 'a') as resf:
                    resf.write('[+] used time: [%d:%02d]'%(usedtime / 60, usedtime % 60))
                    resf.write('ext time: [%d:%02d]'%(needtime / 60, needtime % 60))
                    resf.write('step:[', i, ']loss:[', loss__, ']\n')
                '''
                
                imgpath = 'result/temperature_%s/image'%(version) + version + '/step_' + str(i) + '/'
                if not os.path.exists(imgpath):
                    os.makedirs(imgpath)
                SaveImage(np.array(res[0]).reshape((26, 20, 20)) * 30., imgpath + 'p')
                SaveImage(np.array(one_batch[-1][0]).reshape((26, 20, 20)) * 30., imgpath + 'y')
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
                
    predictresult = []
    print("smallest loss:", min_loss)
    if isPredicting:
        print('Predicting')
        lenP = lenPredict * monthinter
        maxseqneed = max(intervals) * max(channels)
        alldata = alldata[:maxseqneed]
        for i in range(1, lenP + 1):
            lastdata = get_lastest_batch(channels, intervals, alldata)
            indict = {}
            for j in range(len(x_input)):
                indict[x_input[j]] = lastdata[j]
            
            res = sess.run([resnet], indict)
            res = res[0].reshape((1, 26, 20, 20))
            predictresult.append(res[0])
            if i % monthinter == 0:
                print('predicted', int(i/monthinter), '....')
                imgpath = 'result/temperature_%s/image'%(version) + version + '/future_' + str(int(i / monthinter)) + '/'
                if not os.path.exists(imgpath):
                    os.makedirs(imgpath)
                SaveImage(res[0].reshape((26, 20, 20)) * 30., imgpath)
            #print(res[0].shape)
            #print(alldata.shape)
            #print(res.shape)
            alldata = np.vstack([res, alldata])
            #print(alldata.shape)
        predictresult = np.array(predictresult)[::monthinter]
        print(predictresult.shape)
        matpath = 'result/temperature_%s/'%(version) + 'matrix/'
        print(matpath)
        if not os.path.exists(matpath):
            os.makedirs(matpath)
        savemat(matpath + 'predict_result.mat', {'data': predictresult})
