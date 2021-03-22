#!/usr/bin/python
# coding = utf-8
#This module is part of an analysis package

Authorinfo = """
             ------------------Name: Yuan-Chao Hu--------------
             --------------Email: ychu0213@gmail.com-----------
             ----------Web: https://yuanchaohu.github.io/------
             """

Docstr = """
         This module calculates gyration tensor 
         which is a tensor that describes the second moments of posiiton
         of a collection of particles

         gyration tensor is a symmetric matrix of shape (ndim, ndim)

         ref: https://en.wikipedia.org/wiki/Gyration_tensor
         """

import numpy as np 

def Gyration_tensor(groupofatoms):
    """calculate gyration tensor for three dimensional systems

    a group of atoms should be first defined
    groupofatoms are the original coordinates of the selected group
    of a single configuration

    the atom coordinates of the cluster should be removed from PBC
    which can be realized by ovito 'cluster analysis' method 
    by choosing 'unwrap particle coordinates'
    """

    ndim = 3
    results = np.zeros((ndim, ndim))
    conbinations = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)] 
    #0, 1, 2 = x, y, z

    #-----the origin of the coordinate system has been chosen such
    #that the center of mass is (0 ,0 ,0)
    centerofmass = groupofatoms.mean(axis = 0)[np.newaxis, :]
    groupofatoms = groupofatoms - centerofmass

    atomN = groupofatoms.shape[0]
    print ('-----------number of atoms: %d------------'%atomN)
    #-------calculate the tensor--------
    for (m, n) in conbinations:
        Smn = 0
        for i in range(atomN):
            Smn += groupofatoms[i, m] * groupofatoms[i, n]
        results[m, n] = Smn / atomN
        results[n, m] = Smn / atomN

    #-------calculate shape descriptors----------
    w, _ = np.linalg.eig(results)
    principalcomponents = np.sort(w)

    radiusofgyration = np.sqrt(principalcomponents.sum())
    asphericity      = 1.5 * principalcomponents[2] - 0.5 * principalcomponents.sum()
    acylindricity    = principalcomponents[1] - principalcomponents[0]
    shapeanisotropy  = (asphericity ** 2 + 0.75 * acylindricity ** 2) / radiusofgyration**4
    dimensionality   = np.log10(atomN) / np.log10(radiusofgyration)

    print ('---returned: radiusofgyration, asphericity, acylindricity, shapeanisotropy, fractal dimensionality----')
    return radiusofgyration, asphericity, acylindricity, shapeanisotropy, dimensionality