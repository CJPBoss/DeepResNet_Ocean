import numpy as np
from scipy.io import loadmat

def CreateImage(path):
    return np.swapaxes(loadmat(path)['temp'], 0, 2)[::, ::-1, ::]

if __name__ == '__main__':
    print('[!] test read BOA_Argo_.mat ......')
    from PIL import Image
    def SaveImage(image, name=None):
        def processpixel(num):
            '''
                map (0, 30) to [0, 255], and NaN to 0
            '''
            if num != num:
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
                img.save('temp/' + str(i) + '.png')
            else:
                img.save('temp/' + name + str(i) + '.png')    
    import time
    
    t1 = time.time()
    image = CreateImage(r'E:\Code\summer\octave\mat\BOA_Argo_2004_05.mat')
    t2 = time.time()
    
    print('[+] finish load data from .mat, time [%f]' % (t2 - t1))
    SaveImage(image)
    print('[!] finish test ......')