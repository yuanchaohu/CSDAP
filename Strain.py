#!/usr/bin/python
# coding = utf-8
#This module is part of an analysis package

Authorinfo = """
             ------------------Name: Yuan-Chao Hu--------------
             --------------Email: ychu0213@gmail.com-----------
             ----------Web: https://yuanchaohu.github.io/------
             """

Docstr = """
         This module calculates Von Mises Strain to detect the local non-affine deformation
         Ref. Ju Li et al. Materials Transactions 48, 2923 (2007)

         The code accounts for both orthogonal and triclinic boxes by using h-matrix to 
         deal with periodic boundary conditions. This is important in dealing with
         particle distance in triclinic cells
         """

import numpy  as np 
import pandas as pd 
from dump import readdump
import os 
from ParticleNeighbors import Voropp

def Vonmises(inputfile, Neighborfile, ndim, strainrate, ppp = [1,1,1], dt =0.002, filetype = 'lammps', moltypes = '', outputfile = ''):
    """ Calculate Non-Affine Von-Mises Strain (local shear invariant)
        
        With the first snapshot of inputfile as reference
        The unit of strianrate should in align with the intrinsic time unit (i.e with dt)
        The code accounts for both orthogonal and triclinic boxes

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

    d = readdump(inputfile, ndim, filetype, moltypes)
    d.read_onefile()
    positions      = d.Positions
    particlenumber = d.ParticleNumber[0]
    snapshotnumber = d.SnapshotNumber
    boxlength      = d.Boxlength
    hmatrix        = d.hmatrix
    timestep       = d.TimeStep[1] - d.TimeStep[0] 
    PI             = np.eye(ndim, dtype = int) #identity matrix along diag 
    results        = np.zeros((particlenumber, snapshotnumber - 1))

    fneighbor      = open(Neighborfile, 'r')
    Neighborlist   = Voropp(fneighbor, particlenumber) #neighbor list [number, list...]
    fneighbor.close()
    for i in range(particlenumber):
        neighbors = Neighborlist[i, 1: Neighborlist[i, 0] + 1]
        RIJ0      = positions[0][neighbors] - positions[0][i]   #reference snapshot: the initial one
        #periodic  = np.where(np.abs(RIJ0 / boxlength[0]) > 0.50, np.sign(RIJ0), 0)
        #RIJ0     -= boxlength[0] * periodic * ppp #remove periodic boundary conditions
        matrixij  = np.dot(RIJ0, np.linalg.inv(hmatrix[0]))
        RIJ0      = np.dot(matrixij - np.rint(matrixij) * ppp, hmatrix[0])
        for j in range(snapshotnumber - 1):
            RIJ1     = positions[j + 1][neighbors] - positions[j + 1][i] #deformed snapshot
            matrixij = np.dot(RIJ1, np.linalg.inv(hmatrix[j + 1]))
            RIJ1     = np.dot(matrixij - np.rint(matrixij) * ppp, hmatrix[j + 1])
            PJ       = np.dot(np.linalg.inv(np.dot(RIJ0.T, RIJ0)), np.dot(RIJ0.T, RIJ1))
            etaij    = 0.5 * (np.dot(PJ, PJ.T) - PI)
            results[i, j] = np.sqrt(0.5 * np.trace(np.linalg.matrix_power(etaij - (1 / ndim) * np.trace(etaij) * PI, 2)))
    
    results = np.column_stack((np.arange(particlenumber) + 1, results))
    strain  = np.arange(snapshotnumber) * timestep * dt * strainrate
    results = np.vstack((strain, results))
    names   = 'id   The_first_row_is_the_strain.0isNAN'
    fformat = '%d ' + '%.6f ' * (snapshotnumber - 1)
    if outputfile:
        np.savetxt(outputfile, results, fmt = fformat, header = names, comments = '')

    print ('------ Calculate Von Mises Strain Over -------')
    return results, names