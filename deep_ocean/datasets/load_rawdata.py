import numpy as np
import json

from scipy.io import loadmat, savemat
from scipy import interpolate

depthkey = 'pres'
tempkey = 'temp'
depthc = 300
depran = 3
latsta = 87
latran = 3
lonsta = 262
lonran = 3
default_conf = {'depth':300, 'lat_range':[latsta, latsta + latran], 'lon_range':[lonsta, lonsta + lonran]}

def interdepth(depseq, map, conf):
    depth_inter_func = interpolate.interp1d(depseq, map, kind='cubic', axis=0)
    new_depseq = range(0, conf['depth'] + 1, conf['interval_depth'])
    interval_map = depth_inter_func(new_depseq)
    return interval_map

def getareamap(orgdata, conf):
    raw_map = orgdata[tempkey]
    depth_lat_lon_map = np.swapaxes(raw_map, 0, 2)
    depr = conf['dep_range']
    latr = conf['lat_range']
    lonr = conf['lon_range']
    area_map = depth_lat_lon_map[depr[0]:depr[1], latr[1]:latr[0]:-1, lonr[0]:lonr[1]]
    return area_map

def getdepthrange(orgdata, conf):
    depseq = orgdata[depthkey].flatten()
    i = 0
    l = len(depseq)
    while i < l and depseq[i] <= conf['depth']:
        i += 1
    tar_depseq = depseq[:i]
    return tar_depseq
    
    
def load_mat(matpath, sconf=None):
    if sconf is None:
        sconf = default_conf
    orgdata = loadmat(matpath)
    depseq = getdepthrange(orgdata, sconf)
    #print(depseq)
    sconf['dep_range'] = [len(depseq) - sconf['depth_range'], len(depseq)]
    area_map = getareamap(orgdata, sconf)
    return area_map
    #inter_map = interdepth(depseq, area_map, sconf)
    
    #return inter_map
    '''
    area_map = getarray(path, 'temp', depr, latr, lonr)
    interval_5m_depth_map = interpolate_depth(pres, area_map)
    '''
    
if __name__ == '__main__':
    path = r'G:\Argo\BOA_Argo_2004\BOA_Argo_2004_01.mat'
    a = load_mat(path)
    savemat("temp.mat", {'temp':a})
    '''
    path = r'/home/Deep_Ocean/temp/original_data/'
    map = load_mat(path)
    timee = 20
    nx = 20 * timee
    x = range(0, nx, timee)
    func = interpolate.interp1d(x, map, kind='cubic', axis=1)
    x1 = range(0, nx - timee)
    map = func(x1)
    func = interpolate.interp1d(x, map, kind='cubic', axis=2)
    x1 = range(0, nx - timee)
    map = func(x1)
    
    import ReadBOA_Argo_mat
    ReadBOA_Argo_mat.SaveImage(map)
    '''
