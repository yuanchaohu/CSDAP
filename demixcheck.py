#coding = utf-8

Authorinfo = """
             ------------------Name: Yuan-Chao Hu--------------
             --------------Email: ychu0213@gmail.com-----------
             ----------Web: https://yuanchaohu.github.io/------
             """

Docstr = """
             calculate the probability of atom A in the first neighbor of atom B
         """

import numpy as np 
import os 
from   dump import readdump
from   ParticleNeighbors import Voropp

def neighbortypes(inputfile, ndim, neighborfile, filetype = 'lammps', moltypes = '', outputfile = ''):
    """Analysis the fractions of atom A in the first neighbor of atom B
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

    #get the coordinate information
    d = readdump(inputfile, ndim, filetype, moltypes)
    d.read_onefile()
    #get the neighbor list from voronoi analysis
    fneighbor = open(neighborfile, 'r')

    results = np.zeros((d.SnapshotNumber, 3)) #for binary system 11 12/21 22
    for i in range(d.SnapshotNumber):
        neighborlist = Voropp(fneighbor, d.ParticleNumber[i])  #neighbor list [number, list....]
        neighbortype = d.ParticleType[i]

        medium = np.zeros(6)
        for j in range(d.ParticleNumber[i]):
            neighborsij = neighborlist[j, 1 : (neighborlist[j, 0] + 1)]
            data11 = (neighbortype[j] + neighbortype[neighborsij] == 2).sum()
            if data11 > 0:
                medium[0] += neighborlist[j, 0]
                medium[1] += data11
            
            data12 = (neighbortype[j] + neighbortype[neighborsij] == 3).sum()
            if data12 > 0:
                medium[2] += neighborlist[j, 0]
                medium[3] += data12

            data22 = (neighbortype[j] + neighbortype[neighborsij] == 4).sum()
            if data22 > 0:
                medium[4] += neighborlist[j, 0]
                medium[5] += data22

        results[i, 0] = medium[1] / medium[0]
        results[i, 1] = medium[3] / medium[2]
        results[i, 2] = medium[5] / medium[4]

    fneighbor.close() 
    if outputfile:
        names = '11  12/21  22'
        np.savetxt(outputfile, results, fmt = 3 * ' %.6f', header = names, comments = '')

    print ('-------demix checking over------')
    return results, names