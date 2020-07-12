#!/usr/bin/python
#coding = utf-8
#This module is part of an analysis package

Authorinfo = """
             ------------------Name: Yuan-Chao Hu--------------
             --------------Email: ychu0213@gmail.com-----------
             ----------Web: https://yuanchaohu.github.io/------
             """

Docstr = """
        This module is used to do coarse graining of a particle-level
        property over the given neighbor list
"""

import numpy as np 
import pandas as pd 
from dump import readdump
from ParticleNeighbors import Voropp

def CG(ordering, neighborfile, outputfile):
    """Coarse Graining over of ordering over cetain neighbor list
    
    ordering: input array of the atomic property to be coarse-grained
    it should have the shape of [num_of_atom, num_of_snapshot]
    """

    orderingCG = np.zeros_like(ordering) #initiallization
    fneighbor  = open(neighborfile)

    for n in range(ordering.shape[1]):
        dataneigh = Voropp(fneighbor, ordering.shape[0])
        for i in range(ordering.shape[0]):
            indices = dataneigh[i, 1:1+dataneigh[i, 0]].tolist()
            indices.append(i)
            orderingCG[i, n] = ordering[indices, n].mean()

    if outputfile:
        results = np.column_stack((np.arange(ordering.shape[0])+1, orderingCG))
        fmt = '%d ' + '%.10f ' * orderingCG.shape[1]
        np.savetxt(outputfile, results, fmt = fmt, header = 'id order_Coarsegrained', comments = '')

    fneighbor.close()
    print ('-----------Coarse-Graining Done---------')
    return orderingCG