# For this part of the assignment, please implement your own code for all computations,
# Do not use inbuilt functions like fft from either numpy, opencv or other libraries

import numpy as np
import cmath
import scipy
class DFT:

    def forward_transform(self, matrix):
        """Computes the forward Fourier transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a complex matrix representing fourier transform"""
        print("--------FOURIER TRNASFORM----")
        print(np.fft.fft2(matrix))

        N = len(matrix)

        darray = [[0.0 for i in range(N)] for j in range(N)]

        for u in range(N):
            for v in range(N):
                sums = 0.0

                for i in range(N):
                    for j in range(N):

                        val = matrix[i,j]
                        e = cmath.exp(- 1j * (2.0*cmath.pi) * ((float(u * i) / N) + (float(v * j) / N)))
                        sums += val * e

                darray[u][v] = sums


        return darray

    def inverse_transform(self, matrix):
        """Computes the inverse Fourier transform of the input matrix
        matrix: a 2d matrix (DFT) usually complex
        takes as input:
        returns a complex matrix representing the inverse fourier transform"""
        print("--------INVERSE---------")
        print(np.fft.ifft2(matrix))

        N = len(matrix)

        darray = [[0.0 for i in range(N)] for j in range(N)]

        for i in range(N):
            for j in range(N):
                sums = 0.0

                for u in range(N):
                    for v in range(N):
                        val = matrix[u][v]
                        e = cmath.exp(1j * (2.0 * cmath.pi) * ((float(u * i) / N) + (float(v * j) / N)))
                        sums += (1/(N*N)) * val * e

                darray[i][j] = sums


        """
        for i in range(N):
            for j in range(N):
                sums = 0.0

                for u in range(N):
                    for v in range(N):
                        val = matrix[u][v]
                        e = cmath.cos(((2.0 * cmath.pi)/N) * (float(u * i))) + 1j*cmath.sin(((2.0 * cmath.pi)/N) * ( (float(v * j))))
                        sums += (1 / (N * N)) * val * e

                darray[i][j] = sums
        """
        return darray


    def discrete_cosine_tranform(self, matrix):
        """Computes the discrete cosine transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing discrete cosine transform"""
        print("--------DCT----------")
        print(matrix)
        N = len(matrix)

        darray = [[0.0 for i in range(N)] for j in range(N)]

        for u in range(N):
            for v in range(N):
                sums = 0.0

                for i in range(N):
                    for j in range(N):
                        val = matrix[i, j]
                        e = cmath.cos(((2.0 * cmath.pi)/N) * (float(u * i) + float(v * j)))
                        sums += val * e

                darray[u][v] = sums

        return darray

    def magnitude(self, matrix):
        """Computes the magnitude of the DFT
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing magnitude of the dft"""
        print("-------MAG-------")

        N = len(matrix)

        for i in range(N):
            for j in range(N):
                matrix[i][j] = abs(matrix[i][j])

        return matrix