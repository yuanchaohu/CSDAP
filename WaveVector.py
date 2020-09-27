#!/usr/bin/python
# coding = utf-8
#This module is part of an analysis package

Authorinfo = """
             ------------------Name: Yuan-Chao Hu--------------
             --------------Email: ychu0213@gmail.com-----------
             ----------The University of Tokyo (2020)----------
             """
             
Docstr = """
         This module is used to generate wave-vector for calculations
         like static/dynamic structure factor
         """

import os
import numpy  as np 
from math import pi 

def wavevector3d(Numofq = 500):
    """ Define Wave Vector for Structure Factors at three dimension """

    wavenumber = np.square(np.arange(Numofq))
    wavevector = []
    for a in range(Numofq):
        for b in range(Numofq):
            for c in range(Numofq):
                d = a**2 + b**2 + c**2
                if d in wavenumber: 
                    wavevector.append(np.array([d, a, b, c]))
    wavevector = np.ravel(np.array(wavevector))[4:].reshape((-1, 4))
    wavevector = wavevector[wavevector[:, 0].argsort()]
    return wavevector

def wavevector2d(Numofq = 500):
    """ Define Wave Vector for Structure Factors at two dimension """

    wavenumber = np.square(np.arange(Numofq))
    wavevector = []
    for a in range(Numofq):
        for b in range(Numofq):
            d = a**2 + b**2
            if d in wavenumber: 
                wavevector.append(np.array([d, a, b]))
    wavevector = np.ravel(np.array(wavevector))[3:].reshape((-1, 3))
    wavevector = wavevector[wavevector[:, 0].argsort()]   
    return wavevector

def choosewavevector(Numofq, ndim):
    """ Choose Wavevector in dynamic structure factor """

    if ndim == 3:
        return wavevector3d(Numofq)
    elif ndim == 2:
        return wavevector2d(Numofq)
