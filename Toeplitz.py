import numpy as np
from scipy.fft import fft, ifft, fftn, ifftn, rfft, irfft, rfftn, irfftn
from timeit import timeit
from numba import jit

import matplotlib.pyplot as plt
#from memory_profiler import profile

class Toeplitz:
    #@profile
    def __init__(self, values, P, QX, R, num_processes=1, f=1):
        if f > 1 and values.shape[-1] != values.shape[-2]:
            raise ValueError('"Interior" matrix must be square.')
        self.f = f # Size of the final dense ("interior") matrix
        self.values = values # np.array of shape (2*n1-1, ..., 2*nM-1, f, f)
                            # or (2*n1-1, ..., 2*nM-1) if f=1
        self.M = values.ndim - (0 if f == 1 else 2) # Number of blocks levels
        for s in values.shape[:self.M]:
            if not s % 2:
                raise ValueError("The number of blocks defining each level of a multilevel block-Toeplitz matrix must be odd.")
        self.T = tuple([(i + 1)//2 for i in values.shape[:self.M]]) # Tuple of dimensions of blocks at each level (n1, ... nM)
        self.dim = np.prod(np.array(self.T)) * f # Размерность квадратной Тёплицевой матрицы, которую мы храним в сжатом виде
        self.P = P
        self.R = R
        self.QX = QX
        self.num_processes = num_processes
        print('Toeplitz init 1')
        if self.M == 2:
            self.shifted_list = [-(self.T[0] - 1), -(self.T[1] - 1)]
            self.arange_selfM = np.arange(self.M)
            self.arT0 = np.arange(self.T[0])
            self.arT1 = np.arange(self.T[1])
            self.ix = np.ix_(self.arT0, self.arT1) #конструирует сетку из индексов
        elif self.M == 1:
            self.shifted_list = [-(self.T[0] - 1)]
            self.arange_selfM = np.arange(self.M)
            self.arT0 = np.arange(self.T[0])
            self.ix = np.ix_(self.arT0)
        elif self.M == 3:
            self.shifted_list = [-(self.T[0] - 1), -(self.T[1] - 1), -(self.T[2] - 1)]
            self.arange_selfM = np.arange(self.M)
            self.arT0 = np.arange(self.T[0])
            self.arT1 = np.arange(self.T[1])
            self.arT2 = np.arange(self.T[2])
            self.ix = np.ix_(self.arT0, self.arT1, self.arT2)
        #self.fft_mat = fftn(np.roll(self.values[::-1], self.shifted_list, axis=self.arange_selfM), workers=self.num_processes)
        self.fft_mat = fftn(np.roll(self.values[::-1], self.shifted_list, axis=self.arange_selfM))
        print('Toeplitz init finished')

    def fullmatr(self):
        if self.f == 1 and self.M == 1:
            shp = (self.T[0], self.T[0])
            n = self.values.strides[0]
            return np.lib.stride_tricks.as_strided(self.values[self.T[0]-1:], shape=shp, strides=(-n, n)).copy()
        elif self.f > 1 and self.M == 1:
            return np.vstack([np.hstack([self.values[j] for j in range(i, i + self.T[0])]) for i in range(self.T[0]-1, -1, -1)])
        else:
            sublevel = [Toeplitz(self.values[j], self.P, self.QX, self.R, self.values, f=self.f).fullmatr() for j in range(2 * self.T[0] - 1)]
            return np.vstack([np.hstack([sublevel[j] for j in range(i, i + self.T[0])]) for i in range(self.T[0]-1, -1, -1)])
        return 0

    def __add__(self, y):
        if self.T != y.T or self.f != y.f:
            raise ValueError("Matrices must have the same structure.")
        return Toeplitz(self.values + y.values, f=self.f)

    def __mul__(self, alpha):
        return Toeplitz(alpha * self.values, f=self.f)

    #def vecmul(self, x):
        #if self.f > 1:
        #    raise ValueError("Not implemented yet")
        #if x.shape[0] != self.dim:
        #    raise ValueError("Dimensions of matrix and input vector do not match.")
        #revers = np.flip(self.values)
        #inds = list(map(lambda i: 1 - i, self.T))
        #shifted = np.roll(revers, [-(i - 1) for i in self.T], axis=np.arange(self.M))
        #, s=self.values.shape)

        #ix = np.ix_(*[np.arange(i) for i in self.T])
        #return ifftn(fft_mat * fftn(x.reshape(self.T), s=self.values.shape))[self.ix].real.reshape(-1)
        #return ifftn(fft_mat * fftn(x.reshape(self.T), s=self.values.shape, workers=self.num_processes), workers=self.num_processes)[self.ix].reshape(-1)
        #return ifftn(fft_mat * fftn(x.reshape(self.T), s=self.values.shape))[self.ix].reshape(-1)

    #для собственного алгоритма быстрое умножение матрицы на вектор
    '''def mv(self, v):
        Rv = self.R.dot(v)
        QXv = self.QX.dot(v)
        KQXv = self.vecmul(QXv)
        PKQXv = self.P.dot(KQXv)
        return PKQXv + Rv'''

    #def mv_fast(QXv):


    def mv(self, v):
        fft_vec = fftn(self.QX.dot(v).reshape(self.T), s=self.values.shape, workers=self.num_processes)
        return self.P.dot(ifftn(self.fft_mat * fft_vec, workers=self.num_processes)[self.ix].reshape(-1)) + self.R.dot(v)
        #fft_vec = fftn(self.QX.dot(v).reshape(self.T), s=self.values.shape)
        #return self.P.dot(ifftn(self.fft_mat * fft_vec)[self.ix].reshape(-1)) + self.R.dot(v)
