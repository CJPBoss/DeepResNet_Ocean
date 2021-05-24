from scipy.io import loadmat
import load_rawdata
import numpy as np

def write_one_map(map, file, shape):
    for i in range(shape[0]):
        matrix2d = ''
        for j in range(shape[1]):
            line = ''
            for k in range(shape[2]):
                line += str(map[i][j][k]) + ' '
            line += '\n'
            matrix2d += line
        file.write(matrix2d)
                

def convert(path, filename=None):
    map = load_rawdata.load_mat(path)
    with open(filename + '.csv', 'w') as f:
        shape = map.shape
        print(shape)
        f.write('{0[0]} {0[1]} {0[2]}\n'.format(shape))
        write_one_map(map, f, shape)
        

if __name__ == '__main__':
    path = r'E:\Tim\Argo\data\BOA_Argo_2017\BOA_Argo_2017_07.mat'
    convert(path, 'test')
    