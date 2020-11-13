#!/usr/bin/python
# coding = utf-8
#This module is part of an analysis package

Authorinfo = """
             ------------------Name: Yuan-Chao Hu--------------
             --------------Email: ychu0213@gmail.com-----------
             ----------Web: https://yuanchaohu.github.io/------
             """

Docstr = """
         This module calculates the spatial correlation of a static property
         ref: Cubuk et al., Science 358, 1033–1037 (2017)
         (<S(0)S(r)> - <S>^2)/(<S^2> - <S>^2) [SC_order]
         """

import numpy as np 
from dump import readdump
from ParticleNeighbors import Voropp

def SC_order(inputfile, orderings, ndim = 3, filetype = 'lammps', moltypes = '', rdelta = 0.01, ppp = [1,1,1], outputfile = ''):
    """Calculate spatial correlation of orderings

        inputfile provides atomic coordinates

        orderings is the numpy array of particle-level order
        its shape is [atom_number, snapshot_number]
        this is the standard output format of this analysis package:
        READ: orders = np.loadtxt(orderfile, skiprows = 1)[:, 1:]
    """

    d = readdump(inputfile, ndim, filetype, moltypes)
    d.read_onefile()

    if d.SnapshotNumber != orderings.shape[1]:
        errorinfo = '***inconsistent number of configurations and orderings***'
        raise ValueError(errorinfo)
    if len(orderings.shape) != 2:
        errorinfo = '***change shape of orderings to [num_of_atom, num_of_snapshot]***'
        raise ValueError(errorinfo)

    MAXBIN  = int(d.Boxlength[0].min() / 2.0 / rdelta)
    results = np.zeros((MAXBIN, 2))
    for n in range(d.SnapshotNumber):
        hmatrixinv = np.linalg.inv(d.hmatrix[n])
        for i in range(d.ParticleNumber[n] - 1):
            RIJ      = d.Positions[n][i+1:] - d.Positions[n][i]#[np.newaxis, :]
            matrixij = np.dot(RIJ, hmatrixinv)
            RIJ      = np.dot(matrixij - np.rint(matrixij) * ppp, d.hmatrix[n]) #remove PBC
            distance = np.linalg.norm(RIJ, axis = 1)

            Countvalue, BinEdge = np.histogram(distance, bins = MAXBIN, range = (0, MAXBIN * rdelta))
            results[:, 0] += Countvalue

            orderingsIJ = orderings[i+1:, n] * orderings[i, n]
            Countvalue, BinEdge = np.histogram(distance, bins = MAXBIN, range = (0, MAXBIN * rdelta), weights = orderingsIJ)
            results[:, 1] += Countvalue

    
    results[:, 1] /= results[:, 0]
    medium1 = np.square(orderings.mean())
    medium2 = np.square(orderings).mean()
    results[:, 1] = (results[:, 1] - medium1) / (medium2 - medium1)
    results[:, 0] = (BinEdge[1:] - 0.5 * rdelta)
  
    names   = 'r Sr'
    if outputfile:
        np.savetxt(outputfile, results, fmt = '%.6f', header = names, comments = '')

    print ('-----calculate spatial correlation over----')
    return results, names

def Kronecker_delta(i, j):
    if i == j:
        return 1
    else:
        return 0

def TensorialOrder(file_positions, file_orientations, neighborfile='', ndim=2, filetype='lammps', moltypes='', rdelta=0.02, ppp=[1,1], outputfile=''):
    """
    calculate the correlation function of nematic ordering based on dipoles/spins/ or molecular orientation
    based on the tensorial order parameter Q_ij used for liquid crystal phase
    
    Ref. Zapotocky et al. Phys. Rev. E 51, 1216 (1995)
    """
    
    from dumpAngular import readangular

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
                    medium[i, x, y] = (ndim*mu[x]*mu[y]-Kronecker_delta(x, y))/2
        QIJ.append(medium)
    
    #coarse-graining over certain volume if neighbor list provided
    if neighborfile:
        QIJ0 = QIJ.copy() #keep the original data unchanged during coarse-graining
        f = open(neighborfile)
        for n in range(d1.SnapshotNumber):
            cnlist = Voropp(f, d1.ParticleNumber[n])
            for i in range(d1.ParticleNumber[n]):
                for j in range(cnlist[i, 0]):
                    QIJ[n][i] += QIJ0[n][cnlist[i, 1+j]]
                QIJ[n][i] /= (1+cnlist[i, 0])
        del QIJ0
        f.close()
    
    #calculate spatial correlation
    MAXBIN = int(d1.Boxlength[0].min() / 2.0 / rdelta)
    results = np.zeros((MAXBIN, 2))
    for n in range(d1.SnapshotNumber):
        hmatrixinv = np.linalg.inv(d1.hmatrix[n])
        for i in range(d1.ParticleNumber[n]-1):
            RIJ      = d1.Positions[n][i+1:] - d1.Positions[n][i]
            matrixij = np.dot(RIJ, hmatrixinv)
            RIJ      = np.dot(matrixij-np.rint(matrixij)*ppp, d1.hmatrix[n]) #remove PBC
            distance = np.linalg.norm(RIJ, axis=1)

            Countvalue, BinEdge = np.histogram(distance, bins=MAXBIN, range=(0, MAXBIN*rdelta))
            results[:, 0] += Countvalue

            orderingsIJ = []
            for j in range(i+1, d1.ParticleNumber[n]):
                medium = np.trace(np.matmul(QIJ[n][i], QIJ[n][j]))
                orderingsIJ.append(medium)           
            
            Countvalue, BinEdge = np.histogram(distance, bins=MAXBIN, range=(0, MAXBIN*rdelta), weights=orderingsIJ)
            results[:, 1] += Countvalue

    results[:, 1] /= results[:, 0]
    results[:, 0] = (BinEdge[1:] - 0.5*rdelta)

    names = 'r Cr'
    if outputfile:
        np.savetxt(outputfile, results, fmt='%.6f', header=names, comments='')
    
    print ('--------calculate spatial correlation of nematic order over---------')
    return results, names
