import numpy as np
from scipy.io import loadmat
from PIL import Image

CreateImage = lambda path: np.swapaxes(loadmat(path)['temp'], 0, 2)[::, ::-1, ::]

def SaveImage(image, name=None):
    def processpixel(num):
        '''
            map (0, 30) to [0, 255], and NaN to 0
        '''
        if num != num:
            print('nan')
            return 0
        else:
            a = min([255, int(num * 255 / 30)])
            b = max([0, a])
            return b
            
    shape = image.shape
    img2 = image.flatten()
    img2 = np.array(list(map(processpixel, img2)))
    img2 = img2.reshape(shape)
    
    for i in range(img2.shape[0]):
        img = Image.fromarray(img2[i])
        img = img.convert('L')
        if name == None:
            img.save(str(i) + '.png')
        else:
            img.save(name + str(i) + '.png')
            
if __name__ == '__main__':
    print('[!] test read BOA_Argo_.mat ......')
        
    import time
    
    t1 = time.time()
    image = CreateImage(r'E:\Tim\Argo\data\BOA_Argo_2017\BOA_Argo_2017_07.mat')
    t2 = time.time()
    
    print('[+] finish load data from .mat, time [%f]' % (t2 - t1))
    SaveImage(image)
    print('[!] finish test ......')