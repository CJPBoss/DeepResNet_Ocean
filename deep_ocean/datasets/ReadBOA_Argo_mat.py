import numpy as np
from scipy.io import loadmat
from PIL import Image
import os

CreateImage = lambda path: np.swapaxes(loadmat(path)['temp'], 0, 2)[::, ::-1, ::]

def SaveImage(image, name=None):
    def processpixel(num):
        '''
            map (0, 30) to [0, 255], and NaN to 0
        '''
        if num != num:
            #$print('nan')
            return 0
        else:
            a = min([255, int(num * 255 / 30)])
            b = max([0, a])
            return b
            
    shape = image.shape
    img2 = image.flatten()
    img2 = np.array(list(map(processpixel, img2)))
    img2 = img2.reshape(shape)
    print(shape)
    for i in range(img2.shape[0]):

        #print(type(img2[i]))
        #print(img2[i].shape)
        img = Image.fromarray(np.uint8(img2[i]))
        img = img.convert('L')
        if name == None:
            img.save(str(i) + '.png')
        else:
            img.save(name + str(i) + '.png')
            
if __name__ == '__main__':
    print('[!] test read BOA_Argo_.mat ......')
        
    import time
    
    t1 = time.time()
    #image = CreateImage(r'E:\Tim\Argo\data\BOA_Argo_2017\BOA_Argo_2017_07.mat')
    image = loadmat(r'data_2017.mat')['data']
    t2 = time.time()
    print('[+] finish load data from .mat, time [%f]' % (t2 - t1))
    print(image.shape)
    for m in range(len(image)):
        path = 'm2017/m%02d/' % (m)
        print(path)
        if not os.path.exists(path):
            os.makedirs(path)
        for d in range(len(image)):
            img = Image.fromarray(np.uint8(image[m][d] * 255.0 / 30.0))
            #print(image[m][d] * 255.0 / 30.0)
            img = img.convert('L')
            img.save(path + 'm%02d_d%02d.png'%(m, d))
    print('[!] finish test ......')
