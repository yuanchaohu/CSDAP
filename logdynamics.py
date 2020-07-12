#coding = utf-8

#This module is part of an analysis package

Authorinfo = """
             ------------------Name: Yuan-Chao Hu--------------
             --------------Email: ychu0213@gmail.com-----------
             ----------Web: https://yuanchaohu.github.io/------
             """

Docstr = """
         This module is responsible for calculating particle level dynamics
         Compute self-intermediate scattering functions ISF to test the structural realxation time

         This is suitable for the dump conifgurations by using 'log' output style
         This is realized in LAMMPS by the following command
         
         variable  outlog  equal logfreq2(10,18,10) #(90-10)/18, (900-100)/18, (9000-1000)/18
         dump_modify 1 every v_outlog first yes
         """

import os 
import numpy as np 
import pandas as pd 
from dump import readdump

def alpha2factor(ndim):
    """ Choose factor in alpha2 calculation """

    if ndim == 3:
        return 3.0 / 5.0
    elif ndim == 2:
        return 1.0 / 2.0

def total(inputfile, ndim, filetype = 'lammps', moltypes = '', qmax = 0, a = 1.0, dt = 0.002, ppp = [1,1,1], PBC = True, outputfile = ''):
    """ Compute self-intermediate scattering functions ISF, dynamic susceptibility ISFX4 based on ISF
        Overlap function Qt and its corresponding dynamic susceptibility QtX4
        Mean-square displacements msd; non-Gaussion parameter alpha2
    
        qmax is the wavenumber corresponding to the first peak of structure factor
        a is the cutoff for the overlap function, default is 1.0 (EAM) and 0.3(LJ) (0.3<d>)
        dt is the timestep of MD simulations
        ppp is the periodic boundary conditions
        PBC is to determine whether we need to remove periodic boundary conditions

    """
    print ('-----------------Compute Overall Dynamics of log output--------------------')

    d = readdump(inputfile, ndim, filetype, moltypes)
    d.read_onefile()

    results = np.zeros(((d.SnapshotNumber - 1), 5))
    names  = 't  ISF  Qt  msd  alpha2'

    results[:, 0] = (np.array(d.TimeStep[1:]) - d.TimeStep[0]) * dt

    RII = d.Positions[1:] - d.Positions[0] #only use the first configuration as reference
    if PBC:
        hmatrixinv = np.linalg.inv(d.hmatrix[0])
        for ii in range(RII.shape[0]):
            matrixij = np.dot(RII[ii], hmatrixinv)
            RII[ii]  = np.dot(matrixij - np.rint(matrixij) * ppp, d.hmatrix[0]) #remove PBC

    results[:, 1] = (np.cos(RII * qmax).mean(axis = 2)).mean(axis = 1)
    distance = np.square(RII).sum(axis = 2)
    results[:, 2] = (np.sqrt(distance) <= a).sum(axis = 1) / d.ParticleNumber[0]
    results[:, 3] = distance.mean(axis = 1)
    distance2 = np.square(distance).mean(axis = 1)
    results[:, 4] = alpha2factor(ndim) * distance2 / np.square(results[:, 3])  - 1.0

    if outputfile:
        np.savetxt(outputfile, results, fmt = '%.6f', header = names, comments = '')

    print ('-----------------Compute Overall Dynamics of log output OVER--------------------')
    return results, names

def total_type(inputfile, ndim, filetype = 'lammps', moltypes = '', typeid = 1, qmax = 0, a = 1.0, dt = 0.002, ppp = [1,1,1], PBC = True, outputfile = ''):
    """ Compute self-intermediate scattering functions ISF, dynamic susceptibility ISFX4 based on ISF
        Overlap function Qt and its corresponding dynamic susceptibility QtX4
        Mean-square displacements msd; non-Gaussion parameter alpha2
    
        qmax is the wavenumber corresponding to the first peak of structure factor
        a is the cutoff for the overlap function, default is 1.0 (EAM) and 0.3(LJ) (0.3<d>)
        dt is the timestep of MD simulations
        ppp is the periodic boundary conditions
        typeid is the atom type to calculate the properties
        PBC is to determine whether we need to remove periodic boundary conditions

    """
    print ('-----------------Compute Overall Dynamics of log output--------------------')

    d = readdump(inputfile, ndim, filetype, moltypes)
    d.read_onefile()

    conditions = d.ParticleType[0] == typeid

    results = np.zeros(((d.SnapshotNumber - 1), 5))
    names  = 't  ISF  Qt  msd  alpha2'

    results[:, 0] = (np.array(d.TimeStep[1:]) - d.TimeStep[0]) * dt

    RII = [i[conditions] - d.Positions[0][conditions] for i in d.Positions[1:]]
    RII = np.array(RII)
    if PBC:#remove PBC
        hmatrixinv = np.linalg.inv(d.hmatrix[0])
        for ii in range(RII.shape[0]):
            matrixij = np.dot(RII[ii], hmatrixinv)
            RII[ii]  = np.dot(matrixij - np.rint(matrixij) * ppp, d.hmatrix[0]) 

    results[:, 1] = (np.cos(RII * qmax).mean(axis = 2)).mean(axis = 1)
    distance = np.square(RII).sum(axis = 2)
    results[:, 2] = (np.sqrt(distance) <= a).sum(axis = 1) / conditions.sum()
    results[:, 3] = distance.mean(axis = 1)
    distance2 = np.square(distance).mean(axis = 1)
    results[:, 4] = alpha2factor(ndim) * distance2 / np.square(results[:, 3])  - 1.0

    if outputfile:
        np.savetxt(outputfile, results, fmt = '%.6f', header = names, comments = '')

    print ('-----------------Compute Type Dynamics of log output OVER--------------------')
    return results, names

def chi4(DV, ParticleNumber):
    """
    This function is used to calculate the dynamical sysceptibility 
    from multiple outputs of log dump. Therefore, each time interval
    has several values of the variable, such as Qt, ISF

    DV is the dynamical variable, having shape of [time, sample]
    """

    return ParticleNumber * ((DV**2).mean(axis = 1) - (DV.mean(axis = 1))**2)