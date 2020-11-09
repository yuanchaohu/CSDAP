#!/usr/bin/python
# coding = utf-8
#This module is part of an analysis package

Authorinfo = """
             ------------------Name: Yuan-Chao Hu--------------
             --------------Email: ychu0213@gmail.com-----------
             ----------Web: https://yuanchaohu.github.io/------
             """

Docstr = """
         This module calculates the order parameters for the nematic
         liquid crystal phase
         """

import numpy as np 
from dumpAngular import readangular


def Legendre_polynomials(x):
    return (3 * x**2 - 1) / 2

def Kronecker_delta(i, j):
    if i == j:
        return 1
    else:
        return 0


def OrderParameters(files_orientation, ndim=2):
    """
    calculate the nematic order parameters
    
    1) g2 = <P2(u_i * u_j)>
    2) Q_{alpha, beta}
    """

    d = readangular(files_orientation, ndim)
    d.read_onefile()

    g2 = 0
    Qij_total = 0
    Qij = np.zeros((d.ParticleNumber[0], d.SnapshotNumber))
    for n in range(d.SnapshotNumber):
        #second order Legendre Polynormial
        for i in range(d.ParticleNumber[n]-1):
            uij = (d.velocity[n][i+1:] * d.velocity[n][i]).sum(axis=1)
            g2 += Legendre_polynomials(uij).mean()
        
        #tensorial order
        for i in range(d.ParticleNumber[n]):
            medium = np.zeros((ndim, ndim))
            mu = d.velocity[n][i]
            for x in range(ndim):
                for y in range(ndim):
                    medium[x, y] = 1.50*mu[x]*mu[y]-0.50*Kronecker_delta(x, y)
            Qij_total += medium
            Qij[i, n] = np.linalg.eig(medium)[0].max()

    Qij_total = np.linalg.eig(Qij_total/Qij.size)[0].max()   
    g2 = g2 / d.SnapshotNumber / d.ParticleNumber[0]
    return g2, Qij_total