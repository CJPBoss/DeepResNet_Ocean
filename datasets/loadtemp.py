import numpy as np
from random import randint
import os
from scipy.io import loadmat, savemat
from ReadBOA_Argo_mat import SaveImage

def swapchannel(arr):
    '''
    channel first to channel last
    '''
    l = len(arr.shape) - 1
    for i in range(l):
        arr = arr.swapaxes(i, i+1)
    return np.array(arr)

def swapchannel2(arr):
    '''
    channel last to channel first
    '''
    l = len(arr.shape) - 1
    for i in range(l, 1, -1):
        arr = arr.swapaxes(i, i - 1)
    return np.array(arr)

def _get_date(s): # haven't use yet
    return s[1:].split('_')

factor = 1. / 30   

class DataSet(object):

    def __init__(self, path): # monthly_data_area.mat
        self.__path = path
        self.__load(loadmat(self.__path))   # pass a dict
        self.__ready = False
        self.__mask = [True, True, True]
        
    def __load(self, data):
        self.__datekeys = list(data.keys())[3:] # all the months
        self.__temp = [data[key]*factor for key in self.__datekeys]
    
    def init_cs(self, channels, steps):
        self.__seqdata = self.getseq(channels, steps)
        self.__ready = True
    
    def getdateseq(self, index=-1, channel=3, step=1):
        '''
        get a sequence of monthly data on
        '''
        seq = self.__temp[index::-step]
        if len(seq) == 0:
            print('empty')
            return None
            
        date = [np.array(i) for i in seq[:channel]]
        
        l = len(date)
        if l < channel:
            last = date[-1]
            for i in range(channel - l):
                date.append(np.array(last))
        return swapchannel(np.array(date))
        
    def getseq(self, channels=(3, 3, 3), steps=(1, 3, 12)):
        '''
        return a list with 4 elements,
        0-2 elements are sequence(list) of 4d numpy.array, [depth, weight, height, channel]
        3 element is a sequence(list) of 4d numpy.array, [depth, weight, height, 1]
        '''
        shape = list(self.__temp[0].shape)
        shape.append(1)
        datapool = [[], [], [], [np.array(i).reshape(shape) for i in self.__temp[1:]]]
        self.__len = len(self.__temp)
        datapool[3].append(np.array(self.__temp[-1]).reshape(shape))
        datapool.append(self.__datekeys[1:])
        datapool[4].append(self.__datekeys[-1])
        for i in range(len(channels)):
            if channels[i] is None or channels[i] < 1:
                self.__mask[i] = False
                continue
            for j in range(self.__len):
                #k = self.__len - j - 1
                datapool[i].append(self.getdateseq(index=j, channel=channels[i], step=steps[i]))
        return datapool
    
    #def get_sorted_batch(self, channels=(3, 3, 3), steps=(1, 3, 12)):
        
    
    def next_batch(self, num=100, index=None):
        if self.__ready is False:
            return None
        else :
            batch = [[], [], [], [], []]
            for i in range(num):
                id = index
                if id is None:
                    id = randint(0, self.__len - 1)
                for j in range(3):
                    if self.__mask[j]:
                        batch[j].append(np.array(self.__seqdata[j][id]))
                    else:
                        batch[j] = None
                batch[3].append(np.array(self.__seqdata[3][id]))
                batch[4].append(self.__seqdata[4][id])
            for i in range(4):
                if batch[i] is not None:
                    batch[i] = np.array(batch[i])
                else:
                    batch[i] = None
            return batch
        
    
if __name__ == '__main__':
    path = 'monthly_data_area.mat'
    dataset = DataSet(path) 
    dataset.init_cs(channels=(2, 2, 2), steps=(1, 3, 12))
    a, b, c, y, m = dataset.next_batch(1, 161)
    for i in range(len(a)):
        name = m[i]
        print(name, i)
        if not os.path.exists('temptest/' + name + '2'):
            os.makedirs('temptest/' + name + '2')
        SaveImage(swapchannel2(a[i]).reshape(52, 20, 20), 'temptest/' + name + '2/c_')
        SaveImage(swapchannel2(b[i]).reshape(52, 20, 20), 'temptest/' + name + '2/p_')
        SaveImage(swapchannel2(c[i]).reshape(52, 20, 20), 'temptest/' + name + '2/t_')
        SaveImage(swapchannel2(y[i]).reshape(26, 20, 20), 'temptest/' + name + '2/y_')
    
    #print(dataset.next_batch(1)[0][2])
    '''
    print(dataset.next_batch(1)[0][3][0][0][0])
    print(dataset.next_batch(1)[0][3].shape)
    print(dataset.next_batch(1)[0][3][0][0][0])
    '''

