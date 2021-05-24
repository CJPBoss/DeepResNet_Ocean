import numpy as np
import tensorflow as tf

#from sklearn.metrics import mean_squared_error

from scipy.io import loadmat
path = '/home/lzt/Deep_Ocean'#r'E:\Code\DeepResNet_Ocean__\DeepResNet_Ocean'
import sys
sys.path.append(path)
from datasets.ReadBOA_Argo_mat import SaveImage
depseq = np.array([0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 200, 220, 240, 260, 280, 300])
depindex = [int(i) for i in depseq / 5]
#print(str(depindex)) 
version = 'v1.24'

predata = loadmat('/home/lzt/Deep_Ocean/model/result/temperature_%s/matrix/predict_result.mat'%(version))['data']
preres = []
for m in range(len(predata)):
    sdata = []
    for i in range(len(predata[0])):
        if i in depindex:
            sdata.append(predata[m][i])
    preres.append(np.array(sdata))
preres = np.array(preres)

#preres = loadmat('data_2017.mat')['data']
orgdata2016 = loadmat('data_2016.mat')['data']
orgdata2017 = loadmat('data_2017.mat')['data']

#shape = preres.shape[-1:-4:-1]
#print(shape)
#a = shape[0] * shape[1] * shape[2]
#print('predata', len(predata))
print('orgdata', len(orgdata2017))

'''
y = tf.placeholder(tf.float32, [None, 26, 20, 20])
y_ = tf.placeholder(tf.float32, [None, 26, 20, 20])

loss = tf.losses.mean_squared_error(y, y_)
sess = tf.Session()
'''
for i in range(len(preres)):
    l = np.sqrt(np.mean((orgdata2017[i] - predata[i])**2))#mean_squared_error(orgdata[i], preres[i])
    
    print(i+1,'\t', l)
    #SaveImage()
    
    
'''
1        0.007281018373782305
2        0.006550897351129932
3        0.005878976558836308
4        0.006234953257457228
5        0.006705769140641693
6        0.0075445836159651275
7        0.007311162953081589
8        0.008542282264124974
9        0.009043873011326475
10       0.00903860940714962
11       0.009798791183181756
12       0.011265271146969192
'''
