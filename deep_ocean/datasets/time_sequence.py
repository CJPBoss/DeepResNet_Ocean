import numpy as np
import json
import load_rawdata

from scipy.io import loadmat, savemat
from scipy import interpolate

depthc = 300
depran = 15
latend = 90
latran = 15
lonend = 265
lonran = 15
default_conf = {'depth':300, 'depth_range':depran, 'lat_range':[latend - latran, latend], 'lon_range':[lonend - lonran, lonend]}

mon_days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
def luner_year(year):
    if year % 400 == 0:
        return 1
    elif year % 100 == 0:
        return 0
    elif year % 4 == 0:
        return 1
    return 0

def month_index():
    m_index = []
    id = 0
    for year in range(2004, 2018):
        m_index.append(id)
        id += mon_days[1]
        m_index.append(id)
        id += mon_days[2] + luner_year(year)
        for month in range(3, 13):
            m_index.append(id)
            id += mon_days[month]
    return m_index, id

def create_time_sequence(path, sconf=None):
    filenameformat = 'BOA_Argo_{0}/BOA_Argo_{0}_{1:0>2}.mat'
    fileformat = path + filenameformat
    seq = []
    for year in range(2004, 2018):
        for month in range(1, 13):
            filepath = fileformat.format(year, month)
            print(filepath)
            a = np.array(load_rawdata.load_mat(filepath, default_conf))
            #a = a.reshape([1] + list(a.shape))
            #print([1] + list(a.shape), a.shape)
            seq.append(a)
            
    return np.stack(np.array(seq))
            
    

if __name__ == '__main__':
    #monindex, id = month_index()
    #print(monindex, id)
    #print(len(monindex))
    path = 'G:/Argo/'
    a = create_time_sequence(path)
    print(a.shape)
    savemat('temp10x10.mat', {'temp': a})
    
