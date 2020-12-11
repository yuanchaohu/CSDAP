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

         Ref. Allen & Frenkel Phys. Rev. Lett. 58, 1748 (1987) 
         """

import numpy as np 
from dumpAngular import readangular
from dump import readdump
from ParticleNeighbors import Voropp


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

    return g2, Qij_total[0].max()*2, P2, P2_one #, P2_one#, Qij, P2==P2_one

def g2correlation(file_positions, files_orientations, neighborfile='', ndim=2, filetype='lammps', moltypes='', rdelta=0.02, ppp=[1,1], outputfile=''):
    """
    calculate the spatial correlation version of the second-rank Kirkwood factor
    """

    #read positional information
    d1 = readdump(file_positions, ndim, filetype, moltypes)
    d1.read_onefile()

    #read orientational information
    d2 = readangular(files_orientations, ndim)
    d2.read_onefile()

    if d1.SnapshotNumber != d2.SnapshotNumber:
        raise OSError('*****Positions and Orientations do NOT match*****')
   
    #coarse-grain the atomic orientation if neighbor list provided
    if neighborfile:
        from ParticleNeighbors import Voropp
        vel0 = np.copy(d2.velocity) #keep original data unchanged during coarse-graining
        f = open(neighborfile)
        for n in range(d1.SnapshotNumber):
            cnlist = Voropp(f, d1.ParticleNumber[n])
            for i in range(d1.ParticleNumber[n]):
                for j in range(cnlist[i, 0]):
                    d2.velocity[n][i] += vel0[n][cnlist[i, 1+j]]
                d2.velocity[n][i] /= (1+cnlist[i, 0])
        f.close()
        del vel0
    
    #calculate spatial correlation
    MAXBIN = int(d1.Boxlength[0].min() / 2.0 / rdelta)
    results = np.zeros((MAXBIN, 2))
    for n in range(d1.SnapshotNumber):
        hmatrixinv = np.linalg.inv(d1.hmatrix[n])
        for i in range(d1.ParticleNumber[n]):
            condition = np.arange(d1.ParticleNumber[n], dtype=np.int32) != i

            RIJ      = d1.Positions[n][condition] - d1.Positions[n][i]
            matrixij = np.dot(RIJ, hmatrixinv)
            RIJ      = np.dot(matrixij-np.rint(matrixij)*ppp, d1.hmatrix[n]) #remove PBC
            distance = np.linalg.norm(RIJ, axis=1)

            Countvalue, BinEdge = np.histogram(distance, bins=MAXBIN, range=(0, MAXBIN*rdelta))
            results[:, 0] += Countvalue

            orderingsIJ = (d2.velocity[n][condition] * d2.velocity[n][i]).sum(axis=1)
            orderingsIJ = Legendre_polynomials(orderingsIJ, ndim)         
            Countvalue, BinEdge = np.histogram(distance, bins=MAXBIN, range=(0, MAXBIN*rdelta), weights=orderingsIJ)
            results[:, 1] += Countvalue    

        results[:, 1] /= results[:, 0]
        results[:, 0] = (BinEdge[1:] - 0.5*rdelta)

        names = 'r g2(r)'
        if outputfile:
            np.savetxt(outputfile, results, fmt='%.6f', header=names, comments='')
        
        print ('---------calculate spatial correlation of g2 over--------')
        return results, names


def LocalOrder(file_positions, file_orientations, neighborfile='', ndim=2, filetype='lammps', moltypes='', ppp=[1, 1], eigvals='False', outputfile=''):
    """
    calculate local nematic ordering based on dipoles/spins/ or molecular orientation
    based on the tensorial order parameter Q_ij used for liquid crystal phase
    
    Ref. Zapotocky et al. Phys. Rev. E 51, 1216 (1995)
    """

    #read positional information
    d1 = readdump(file_positions, ndim, filetype, moltypes)
    d1.read_onefile()

    #read orientational information
    d2 = readangular(file_orientations, ndim)
    d2.read_onefile()

    if d1.SnapshotNumber != d2.SnapshotNumber:
        raise OSError('*****Positions and Orienations do NOT match*****')

    #calculate atomic-level tensorial order
    QIJ = []
    for n in range(d1.SnapshotNumber):
        medium = np.zeros((d1.ParticleNumber[n], ndim, ndim))
        for i in range(d1.ParticleNumber[n]):
            mu = d2.velocity[n][i]
            for x in range(ndim):
                for y in range(ndim):
                    medium[i, x, y] = (
                        ndim*mu[x]*mu[y]-Kronecker_delta(x, y))/2
        QIJ.append(medium)

    #coarse-graining over certain volume if neighbor list provided
    if eigvals:
        eigenvalues = np.zeros((d1.ParticleNumber[0], d1.SnapshotNumber))
    traceII = np.zeros((d1.ParticleNumber[0], d1.SnapshotNumber))
    if neighborfile:
        # keep the original data unchanged during coarse-graining
        QIJ0 = np.copy(QIJ)
        f = open(neighborfile)
        for n in range(d1.SnapshotNumber):
            cnlist = Voropp(f, d1.ParticleNumber[n])
            for i in range(d1.ParticleNumber[n]):
                for j in range(cnlist[i, 0]):
                    QIJ[n][i] += QIJ0[n][cnlist[i, 1+j]]
                QIJ[n][i] /= (1+cnlist[i, 0])
                traceII[i, n] = np.trace(np.matmul(QIJ[n][i], QIJ[n][i]))
                if eigvals:
                    eigenvalues[i, n] = np.linalg.eig(QIJ[n][i])[0].max()*2.0               
        del QIJ0
        f.close()
        traceII *= ndim / (ndim - 1)
        traceII = np.sqrt(traceII)

    fmt = '%d ' + ' %.6f' * (traceII.shape[1] - 1)
    traceII = np.column_stack((np.arange(d1.ParticleNumber[0])+1, traceII))
    np.savetxt(outputfile, traceII, fmt=fmt, header='n traceCG', comments='')
    if eigvals:
        eigenvalues = np.column_stack((np.arange(d1.ParticleNumber[0])+1, eigenvalues))
        np.savetxt(outputfile[:-4]+'_eigenvalues.dat', eigenvalues, fmt=fmt, header='n eig', comments='')

    print ('------calculate nematic ordering done------')