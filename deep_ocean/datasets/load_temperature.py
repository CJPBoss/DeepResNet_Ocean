import numpy as np
from scipy.io import loadmat

FILENAME = '/home/lzt/Deep_Ocean/datasets/scaled_v1.05.mat'#'E:\\Code\\DeepResNet_Ocean__\\DeepResNet_Ocean\\datasets\\scaled.mat'
VALNAME = 'data'

def load_data(valname=VALNAME, filename=FILENAME):
    print('[+] loading data...')
    data = loadmat(filename)[valname]
    #print(data.keys())
    #[valname]
    #print(type(data), data.shape)
    data = data[::-1]
    print('[+] loaded data finish.')
    print('----    data [len %d] [shape %s]' % (data.shape[0], data.shape[1:]))
    return np.array(data, dtype=np.float16)
    
def channel_first(tensor):
    '''
    channel last to channel first
    '''
    l = len(tensor.shape) - 1
    for i in range(l, 1, -1):
        tensor = tensor.swapaxes(i, i - 1)
    return np.array(tensor)
    
def channel_last(tensor):    
    '''
    channel first to channel last
    '''
    l = len(tensor.shape) - 1
    for i in range(l):
        tensor = tensor.swapaxes(i, i+1)
    return np.array(tensor)

def create_list(data, index, channel, interval):
    start_index = index
    end_index = start_index + channel * interval
    #print(start_index, end_index)
    res = np.array(data[start_index:end_index:interval])
    while len(res) < channel:
        res = np.vstack((res, res[-1:]))
    return channel_last(res)
    
def get_lastest_batch(channels, intervals, data=None):
    res = create_batch(1, channels, intervals, data, True)
    
    return res[0:-1:1]
    
def create_batch(num, channels, intervals, data=None, lastest=False):
    '''
    num: input size,
    '''
    if data is None:
        data = load_data()
    len_data = len(data)
    len_channel = len(channels)
    
    res = [[] for i in channels]
    res.append([])
    
    y_shape = list(data[0].shape)
    y_shape.append(1)
    y_shape = tuple(y_shape)
    
    for i in range(num):
        index = -1
        if not lastest:
            index = np.random.randint(0, len_data - 1)
        #index = 0
        #print(i, '---', index)
        for j in range(len_channel):
            xprop = create_list(data, index + 1, channels[j], intervals[j])
            res[j].append(xprop)
        res[-1].append(data[index].reshape(y_shape))
    for i in range(len(res)):
        res[i] = np.array(res[i])
    return res
        

if __name__ == "__main__":
    
    data = np.arange(720)
    data = data.reshape((30, 2, 3, 4))
    res = create_batch(5, (2, 3), (4, 5), data)
    for i in range(len(res[0])):
        print(i, '------------------')
        #print(res[-1][i])
        for j in range(len(res)):
            print(j)
            print(res[j][i])
    for i in range(len(res)):
        print(res[i][0].shape)
    '''
    # test create_list pass
    data = np.arange(720)
    data = data.reshape((30, 2, 3, 4))[::]
    for i in range(len(data)):
        print(i, " - \n", data[i])
    print('============')
    a = create_list(data, 3, 4, 3)
    print(a)
    print('------------')
    print(channel_first(a))
    print(a.shape)
    '''
    
