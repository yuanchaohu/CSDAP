#coding = utf-8

Authorinfo = """
             ------------------Name: Yuan-Chao Hu--------------
             --------------Email: ychu0213@gmail.com-----------
             ----------Web: https://yuanchaohu.github.io/------
             """

Docstr = """
             calculate the local density probability profile to see 
             whether there is cavitation
         """

import numpy as np 
import os 
from   dump import readdump

def cavitation(inputfile, ndim = 3, nbin = 5, filetype = 'lammps', moltypes = ''):
    """Count atom number in a small bin to see whether cavitation occurs
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

    if ndim == 3: 
        results = np.zeros((nbin + 1, nbin + 1, nbin + 1))
        for i in range(d.SnapshotNumber):
            #remove the original point to be (0, 0, 0)
            newpositions = d.Positions[i] - d.Boxbounds[i][:, 0]
            binsize = (d.Boxlength[i] / nbin)[np.newaxis, :]
            if i == 0: binvolume = np.prod(binsize) 
            indices, counts = np.unique(np.rint(newpositions / binsize), axis = 0, return_counts = True)
            indices = indices.astype(np.int)
            for j in range(indices.shape[0]):
                results[indices[j][0], indices[j][1], indices[j][2]] += counts[j]

        results = results / d.SnapshotNumber
        #-----return empty bin number------------------
        return results.size - np.count_nonzero(results)
