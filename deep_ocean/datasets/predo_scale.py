import numpy as np
from scipy.io import savemat, loadmat

FILEPATH = 'scaled.mat'

def loaddata(valname, filepath):
    print('[+] loading data...')
    data = loadmat(filepath)
    temp = data[valname]
    print('[+] loaded finish')
    return np.array(temp)
    
def scaleimage(image):
    def processpixel(num):
        return num / 30
    print('[+] scaling data...')
    shape = image.shape
    img2 = image.flatten()
    img2 = np.array(list(map(processpixel, img2)))
    print('[+] scaled finish')
    return img2.reshape(shape)

if __name__ == '__main__':
    data = loaddata('data', FILEPATH)
    image = scaleimage(data)
    savemat("scaled.mat", {'data': image})
