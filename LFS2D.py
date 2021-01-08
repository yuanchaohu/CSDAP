#!/usr/bin/python
# coding = utf-8
#This module is part of an analysis package

Authorinfo = """
             ------------------Name: Yuan-Chao Hu--------------
             --------------Email: ychu0213@gmail.com-----------
             ----------Web: https://yuanchaohu.github.io/------
             """

Docstr = """
         This module pick up local structures based on the dipole/
         spin orientation
         """

import numpy as np 
from dump import readdump
from dumpAngular import readangular
from ParticleNeighbors import Voropp

def Pentagon(file_positions, file_orientations, neighborfile, ppp=[1,1], tol=18, filetype='lammps', moltypes='', outputfile=''):
    """
    choose a paticle based on its orientation W.S.T its neighbors
    tol is the angle tolerance in degree

    only two nearest neighbors are required
    """

    #read positions
    d1 = readdump(file_positions, 2, filetype, moltypes)
    d1.read_onefile()
   
    #read orientational information
    d2 = readangular(file_orientations, 2)
    d2.read_onefile()
    #get the unit vector
    vectors = [u/np.linalg.norm(u, axis=1)[:, np.newaxis] for u in d2.velocity]

    fneighbor = open(neighborfile, 'r')
    positional = np.zeros((d1.ParticleNumber[0], d1.SnapshotNumber), dtype=np.int32)
    orientational = np.zeros_like(positional)
    
    for n in range(d1.SnapshotNumber):
        Neighborlist = Voropp(fneighbor, d1.ParticleNumber[n])
        hmatrixinv = np.linalg.inv(d1.hmatrix[n])
        for i in range(d1.ParticleNumber[n]):
            ineighbors = Neighborlist[i, 1:1+Neighborlist[i, 0]]
            #orientational
            COSs = (vectors[n][i] * vectors[n][ineighbors]).sum(axis=1)
            angles = np.degrees(np.arccos(COSs))
            condition = np.abs(angles - 72) < tol
            if condition.all():
                orientational[i, n] = 1
            
            #positional
            RIJ = d1.Positions[n][ineighbors] - d1.Positions[n][i]
            matrixij = np.dot(RIJ, hmatrixinv)
            RIJ = np.dot(matrixij-np.rint(matrixij)*ppp, d1.hmatrix[n])  # remove PBC
            distance = np.linalg.norm(RIJ, axis=1)
            COSs = np.dot(RIJ[0], RIJ[1]) / np.prod(distance)
            angles = np.degrees(np.arccos(COSs))
            condition = np.abs(angles - 108) < tol
            if condition.all():
                positional[i, n] = 1
        
    fneighbor.close()
    results = positional * orientational
    results = np.column_stack((np.arange(d1.ParticleNumber[0])+1, results))
    if outputfile:
        names = 'id pentagon'
        np.savetxt(outputfile, results, fmt='%d', header=names, comments='')
    
    print ('--------picking up pentagons done----------')
    
