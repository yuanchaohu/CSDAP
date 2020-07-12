#!/usr/bin/python
# coding = utf-8
#This module is part of an analysis package

Authorinfo = """
             ------------------Name: Yuan-Chao Hu--------------
             --------------Email: ychu0213@gmail.com-----------
             ----------Web: https://yuanchaohu.github.io/------
             """

Docstr = """
         This module calculates local tetrahedral order in 3D

         First step: select the 4-nearest neighbors
         Second step: calcultae local tetrahedral order over these neighbors
         """

import numpy as np 
from   dump   import readdump

def local_tetrahedral_order(dumpfile, filetype = 'lammps', moltypes = '', ppp = [1,1,1], outputfile = ''):
    """Calcultae local tetrahedral order parameter
        The keyword filetype is used for different MD engines
        It has four choices:
        'lammps' (default)

        'lammpscenter' (lammps molecular dump with known atom type of each molecule center)
        moltypes is a dict mapping center atomic type to molecular type
        moltypes is also used to select the center atom 
        such as moltypes = {3: 1, 5: 2}

        'gsd' (HOOMD-blue standard output for static properties)
    
        'gsd_dcd' (HOOMD-blue outputs for static and dynamic properties)
    """

    d = readdump(dumpfile, 3, filetype, moltypes)
    d.read_onefile()
    num_nearest = 4  #number of selected nearest neighbors
    results = np.zeros((max(d.ParticleNumber), d.SnapshotNumber))

    for n in range(d.SnapshotNumber):
        hmatrixinv = np.linalg.inv(d.hmatrix[n])
        Positions  = d.Positions[n]
        for i in range(d.ParticleNumber[n]):
            RIJ      = Positions - Positions[i]
            matrixij = np.dot(RIJ, hmatrixinv)
            RIJ      = np.dot(matrixij - np.rint(matrixij) * ppp, d.hmatrix[n]) #remove PBC

            RIJ_norm = np.linalg.norm(RIJ, axis = 1)
            nearests = np.argpartition(RIJ_norm, num_nearest + 1)[:num_nearest + 1]
            nearests = [j for j in nearests if j != i]
            for j in range(num_nearest - 1):
                for k in range(j + 1, num_nearest):
                    medium1 = np.dot(RIJ[nearests[j]], RIJ[nearests[k]])
                    medium2 = RIJ_norm[nearests[j]] * RIJ_norm[nearests[k]]
                    results[i, n] += (medium1 / medium2 + 1.0/ 3) ** 2

    results = np.column_stack((np.arange(d.ParticleNumber[0]) + 1, 1.0 - 3.0 / 8 * results))

    names = 'id q_tetra'
    if outputfile:
        numformat = '%d ' + '%.6f ' * (results.shape[1] - 1)
        np.savetxt(outputfile, results, fmt = numformat, header = names, comments = '')

    print ('---------calculate local tetrahedral order over---------')
    return results, names