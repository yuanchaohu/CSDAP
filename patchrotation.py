#!/usr/bin/python
#coding = utf-8
#This module is part of an analysis package

Authorinfo = """
             ------------------Name: Yuan-Chao Hu--------------
             --------------Email: ychu0213@gmail.com-----------
             ----------Web: https://yuanchaohu.github.io/------
             """

Docstr = """
        This module is used to calculate local rotational degree of 
        each patchy particle with respect to its neighbors
"""

import numpy as np 
import pandas as pd 
from dump import readdump

def cal_vector(filename, num_patch = 12, ndim = 3, ppp = [1,1,1]):
    """
    calculate the rotational vectors of each particle considering each patch
    """

    #-----get configuration information-----
    d = readdump(filename, ndim)
    d.read_onefile()
    num_atom = [int(i / (num_patch + 1)) for i in d.ParticleNumber]

    #-----get vector information-------
    fdump   = open(filename, 'r')
    pos_all = [] #list of center and patch positions
    for n in range(d.SnapshotNumber):
        medium = np.zeros((num_atom[n], int(1+num_patch), ndim)) 
        #three dimensional array for both center and patch positions
        #the first is the center
        for i in range(9):
            fdump.readline()
        for i in range(num_atom[n]): #also the number of blocks
            for j in range(1+num_patch):
                medium[i, j] = [float(ii) for ii in fdump.readline().split()[2: 2+ndim]]
            
            #-----remove PBC-------
            halfbox  = (d.Boxlength[n] / 2.0)[np.newaxis, :]
            RIJ      = medium[i, 0][np.newaxis, :] - medium[i, 1:]
            periodic = np.where(np.abs(RIJ) > halfbox, np.sign(RIJ), 0).astype(np.int)
            medium[i, 1:] += periodic * d.Boxlength[n][np.newaxis, :]

        pos_all.append(medium)

    print ('--------prepare positions done-------')
    return pos_all, d.SnapshotNumber, num_atom, d.hmatrix

def Rorder(filename, num_patch = 12, ndim = 3, ppp = [1,1,1], neighborfile = '', outputfile = '', outputfileij = ''):
    """calculate local orientational ordering"""

    print ('--------calculate patchy alignment--------')
    from ParticleNeighbors import Voropp
    pos_all, SnapshotNumber, num_atom, hmatrix = cal_vector(filename, num_patch, ndim, ppp)

    fout      = open(outputfileij, 'w')
    fneighbor = open(neighborfile, 'r') #get neighbor list
    results   = np.zeros((num_atom[0], SnapshotNumber))
    for n in range(SnapshotNumber):
        positions = pos_all[n]
        vectors   = []
        for i in range(num_atom[n]):
            medium = positions[i, 1:] - positions[i, 0][np.newaxis, :]
            medium = medium / np.linalg.norm(medium, axis = 1)[:, np.newaxis] #unit vector
            vectors.append(medium) #particle to patch vectors

        fout.write('id cn Rorder_list num_atom = %d\n'%num_atom[n])
        hmatrixinv   = np.linalg.inv(hmatrix[n])
        Neighborlist = Voropp(fneighbor, num_atom[n])
        for i in range(num_atom[n]):
            cnlist   = Neighborlist[i, 1: 1+Neighborlist[i, 0]] #num, list (id-1...)
            RIJ      = positions[cnlist, 0] - positions[i, 0]
            matrixij = np.dot(RIJ, hmatrixinv)
            RIJ      = np.dot(matrixij - np.rint(matrixij) * ppp, hmatrix[n])
            RIJ      = RIJ / np.linalg.norm(RIJ, axis = 1)[:, np.newaxis] #unit vector
            fout.write('%d %d ' %(i+1, Neighborlist[i, 0]))
            for j in range(Neighborlist[i, 0]):
                patch_i = (vectors[i] * RIJ[j]).sum(axis = 1).argmax()
                patch_j = (vectors[cnlist[j]] * RIJ[j]).sum(axis = 1).argmin()
                UIJ = (vectors[i][patch_i] * vectors[cnlist[j]][patch_j]).sum() #U_i * U_j
                results[i, n] += UIJ
                fout.write('%.6f ' %UIJ)
            fout.write('\n')
            results[i, n] = results[i, n] / Neighborlist[i, 0]
    
    fout.close()
    fneighbor.close()

    results = np.column_stack((np.arange(num_atom[0]) + 1, results))
    names = 'id Psi'
    fmt = '%d ' + '%.6f ' * (results.shape[1] - 1)
    np.savetxt(outputfile, results, fmt = fmt, header = names, comments = '')

    print ('--------calculate patchy alignment done--------')
    return results, names