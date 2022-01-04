"""Separate module for any finite-difference/electric field calculations to be accessed from the GUI.
These are functions which should not depend on any GUI class/instance attributes.
"""
import numpy as np
import scipy.ndimage
import scipy.signal

import numba
from numba import jit


def finite_difference(maskarray, potentialarray):
        #use a simple finite-difference process using the average of 4 neighbouring points
        #combine with error tolerance to use the "Jacobi" iteration scheme
        #calculate the numerical values which satisfy Laplace's equation in 2D
        #for the initial boundaries provided
        #the finite-difference equation can be performed as a convolution with a 3x3 kernel of weights for each grid point
        
        conv_factor = 1 * np.array([[0, 1/4, 0], [1/4, 0, 1/4], [0, 1/4, 0]])
        
        v_q_plus = potentialarray.copy()
        while True:
            v_q = v_q_plus.copy()

            v_q_plus[maskarray == True] = scipy.ndimage.convolve(v_q, conv_factor, mode="nearest")[maskarray == True]
#             v_q_plus[maskarray == True] = scipy.signal.convolve(v_q, conv_factor, mode="same", method="fft")[maskarray == True]

            if np.allclose(v_q_plus, v_q, rtol=1e-4):
                break
        
        final_potentials = v_q_plus.copy()
        
        return final_potentials
    
    
@jit(parallel=True)
def sor(maskarray, potentialarray, f=1, rtol=1e-4, boundary="fixed"):
    #successive over-relaxation method
    #using a relaxation parameter "f" to apply weighting to current point vs other points in 5-point stencil
    #f should default to 1 - this is simple Gauss-Seidel; however for quickest results we aim for high f < 2
    #along with Gauss-Seidel selection - using updated points as soon as available
    #speed up using numba.jit - caching the processes to be executed much faster
    #accelerate using parallel execution via numba - numba.prange() replaces python's range()

    v_q_plus = potentialarray.copy()
    ny,nx = v_q_plus.shape
    while True:
        v_q = v_q_plus.copy()

        if boundary == "periodic":
            for k in numba.prange(ny):
                #set edge cases manually to wrap-around to other side of array ("periodic" boundary conditions)
                up, down = k+1, k-1  # - standard values for interior points
                if k == 0:
                    up, down = -1, 1
                if k == ny-1:
                    up, down = -2, 0

                for l in numba.prange(nx):
                    left, right = l-1, l+1
                    if l == 0:
                        left, right = -1, 1
                    if l == nx-1:
                        left, right = -2, 0


                    #apply the 5-point stencil
                    if maskarray[k,l] == True:
                        v_q_plus[k,l] = (1-f) * v_q_plus[k,l] \
                                        + f/4 * (v_q_plus[k,left] + v_q_plus[k,right] + v_q_plus[down,l] + v_q_plus[up,l])


        elif boundary == "fixed":
            #avoid any modification of edge points, process available points without using "ghost" points
            for k in numba.prange(1,ny-1):
                for l in numba.prange(1,nx-1):
                    if maskarray[k,l] == True:
                        up, down = k+1, k-1
                        left, right = l+1, l-1


                        #apply the 5-point stencil
                        v_q_plus[k,l] = (1-f) * v_q_plus[k,l] \
                                        + f/4 * (v_q_plus[k,left] + v_q_plus[k,right] + v_q_plus[down,l] + v_q_plus[up,l])



        #explicit implementation of np.allclose, which seems not to be usable by numba.jit
        #check relative change by all points - if all are below the tolerance level: finish
        if np.less_equal(np.abs(v_q_plus - v_q), rtol * np.abs(v_q)).all():
            break

    return v_q_plus.copy()


def get_Efield(final_potentials):
    #having processed for the numerical values of potential across the grid
    #now interested in getting the electric field shape, E = - grad(V)

    gradV = np.gradient(final_potentials, axis=(1,0))
    Efield = [-gradV[0], -gradV[1]]

    return Efield