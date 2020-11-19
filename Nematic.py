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


def Legendre_polynomials(x, ndim):
    return (ndim * x**2 - 1) / 2

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
    P2_one = 0
    Qij = np.zeros((d.ParticleNumber[0], d.SnapshotNumber))
    for n in range(d.SnapshotNumber):
        #second order Legendre Polynormial
        #for i in range(d.ParticleNumber[n]-1):
            #uij = (d.velocity[n][i+1:] * d.velocity[n][i]).sum(axis=1)
        for i in range(d.ParticleNumber[n]):
            condition = np.arange(d.ParticleNumber[n],dtype=np.int32) != i 
            uij = (d.velocity[n][i] * d.velocity[n][condition]).sum(axis=1)
            g2 += Legendre_polynomials(uij, ndim).sum()
        
        #tensorial order
        Qij_one = 0
        for i in range(d.ParticleNumber[n]):
            medium = np.zeros((ndim, ndim))
            mu = d.velocity[n][i]
            for x in range(ndim):
                for y in range(ndim):
                    medium[x, y] = (ndim*mu[x]*mu[y]-Kronecker_delta(x, y))/2
            Qij_total += medium
            #Qij[i, n] = np.linalg.eig(medium)[0].max()
            Qij_one += medium
        
        Qij_one = np.linalg.eig(Qij_one / d.ParticleNumber[n])
        director = Qij_one[1][Qij_one[0].argmax()]
        medium = (d.velocity[n] * director[np.newaxis, :]).sum(axis=1)
        P2_one += Legendre_polynomials(medium, ndim).mean()


    Qij_total /= Qij.size 
    Qij_total = np.linalg.eig(Qij_total)#[0].max()   
    g2 = g2 / d.SnapshotNumber / d.ParticleNumber[0]
    P2_one /= d.SnapshotNumber
    
    director = Qij_total[1][Qij_total[0].argmax()]
    P2 = 0
    for n in range(d.SnapshotNumber):
        medium = (d.velocity[n] * director[np.newaxis, :]).sum(axis=1)
        P2 += Legendre_polynomials(medium, ndim).mean()
    P2 /= d.SnapshotNumber

    return g2, Qij_total[0].max(), P2 #, P2_one#, Qij, P2==P2_one