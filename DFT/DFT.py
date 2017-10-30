# For this part of the assignment, please implement your own code for all computations,
# Do not use inbuilt functions like fft from either numpy, opencv or other libraries

import numpy as np
import cmath

class DFT:

    def forward_transform(self, matrix):
        """Computes the forward Fourier transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a complex matrix representing fourier transform"""

        """Compute the discrete Fourier Transform of the 1D array x"""
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

    


        return matrix


    def discrete_cosine_tranform(self, matrix):
        """Computes the discrete cosine transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing discrete cosine transform"""



        return matrix


    def magnitude(self, matrix):
        """Computes the magnitude of the DFT
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing magnitude of the dft"""

        return matrix