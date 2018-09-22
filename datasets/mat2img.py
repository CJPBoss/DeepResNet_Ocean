from scipy.io import savemat, loadmat
import numpy as np
from PIL import Image
import os

dirpath = 'E:/Tim/onedrive/Code/py_desktop/DeepResNet_Ocean/datasets/'
filename = 'tdwh.mat'
p = 'temp/'

fpath = dirpath + filename

def simage(image, name=None):
    img = Image.fromarray(image)
    img = img.convert('L')
    if name is None:
        img.save('untitled.png')
    else:
        img.save(name)

def saveimages(images):
    for i in range(len(images)):
        dir = p + str(i) + '/'
        print('\t[+]', i)
        if not os.path.exists(dir):
            os.makedirs(dir)
        for j in range(len(images[i])):
            back = str(j)
            simage(images[i][j], dir + str(j) + '.png')
            

def loaddata(valname='temp', filepath=fpath):
    print('[+] loading data...')
    data = loadmat(filepath)
    temp = data[valname]
    print('[+] loaded finish')
    return np.array(temp)
    
def scaleimage(image):
    def processpixel(num):
        if num != num:
            return 0
        else:
            a = min([255, int(num*255/30)])
            b = max([0, a])
            return b
    print('[+] scaling data...')
    shape = image.shape
    img2 = image.flatten()
    img2 = np.array(list(map(processpixel, img2)))
    print('[+] scaled finish')
    return img2.reshape(shape)
    
def mat2image(valname='temp', filepath=fpath):
    matrix = loaddata(valname, filepath)
    image = scaleimage(matrix)
    saveimages(image)

if __name__ == '__main__':
    mat2image()