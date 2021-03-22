#!/usr/bin/python
# coding = utf-8
#This module is part of an analysis package

import pandas as pd
import numpy as np
from math import sin, cos, pi

Authorinfo = """
             ------------------Name: Yuan-Chao Hu--------------
             --------------Email: ychu0213@gmail.com-----------
             ----------The University of Tokyo (2020)----------
             """

Docstr = """
         This module calculates the Fourier transformation
         of an autocorrelation function by
         Filon's integration method

         ref. https://mathworld.wolfram.com/FilonsIntegrationFormula.html
              http://www.ccl.net/cca/software/SOURCES/FORTRAN/allen-tildesley-book/f.37.shtml
              Allen and Tildesley, Computer Simulation of Liquids, Appendix D 
         """


def Filon_COS(C, t, a=0, outputfile=''):
    """
    C: the auto-correlation function in a numpy array
    t: the time corresponding C in a numpy array
    a: the frequency interval, default is 0
    """

    if len(C)%2 == 0:
        print ('------Warning: number of input data is not odd------')
        C = C[:-1]
        t = t[:-1]
    
    if a == 0: #a is not specified
        a = 2 * pi / t[-1]
    
    Nmax = len(C)
    dt = round(t[1] - t[0], 3)
    if dt != round(t[-1]-t[-2], 3):
        raise ValueError('time is not evenly distributed')

    results = np.zeros((Nmax, 2)) #[omega, FFT]
    for n in range(Nmax):
        omega = n * a 
        results[n, 0] = omega

        #calculate the filon parameters
        theta = omega * dt
        theta2 = theta * theta
        theta3 = theta * theta2
        if theta == 0:
            alpha = 0.0
            beta  = 2.0 / 3.0
            gamma = 4.0 / 3.0
        else:
            alpha = 1.0/theta+sin(2*theta)/2.0/theta2-2.0*sin(theta)*sin(theta)/theta3
            beta  = 2.0*((1+cos(theta)*cos(theta))/theta2-sin(2*theta)/theta3)
            gamma = 4.0*(sin(theta)/theta3-cos(theta)/theta2)
        
        C_even = 0
        for i in range(0, Nmax, 2):
            C_even += C[i]*cos(omega*i*dt)
        C_even -= 0.5 * (C[-1]*cos(omega*t[-1]) + C[0]*cos(omega*t[0]))

        C_odd = 0
        for i in range(1, Nmax-1, 2):
            C_odd += C[i]*cos(omega*i*dt)
        
        results[n, 1] = 2.0*dt*(alpha*(C[-1]*sin(omega*t[-1])-C[0]*sin(omega*t[0]))
                            +   beta*C_even + gamma*C_odd)
    
    results[:, 1] /= pi 
    if outputfile:
        np.savetxt(outputfile, results, fmt='%.6f', header='omega FFT', comments='')

    print ('--------calculate FFT by Filon COSINE method Done-------')
    return results
