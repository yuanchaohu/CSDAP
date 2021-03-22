#!/usr/bin/python
# coding = utf-8
#This module is part of an analysis package

Authorinfo = """
             ------------------Name: Yuan-Chao Hu--------------
             --------------Email: ychu0213@gmail.com-----------
             ----------Web: https://yuanchaohu.github.io/------
             """

Docstr = """
         This module calculates the time correlation of a static property
         The general idea is to calculate 
         [<S(t)S(0)> - <S>**2]/[<S**2> - <S>**2]
         """

import numpy as np 
from dump import readdump
import pandas as pd 


def Orderlife(dumpfile, orderings, ndim=3, dt=0.002, outputfile='', filetype='lammps', moltypes=''):
    """Calculate the time decay of structural ordering

    orderings is the numpy array of structural ordering
    its shape is [atom_number, snapshot_number]
    this is the standard output format of this analysis package:
    READ: orderings = np.loadtxt(orderfile, skiprows = 1)[:, 1:]

    dumpfile is used to get the time scale
    """

    print ('------------Calculate Order time self-correlation---------')
    d = readdump(dumpfile, ndim, filetype, moltypes)
    d.read_onefile()

    TimeStep = d.TimeStep[1] - d.TimeStep[0]
    if TimeStep != d.TimeStep[-1] - d.TimeStep[-2]:
        print ('Warning: ********time interval changes************')

    num_config = orderings.shape[1]
    results = np.zeros((num_config - 1, 2))
    names = 't  St'
    cal_SIt = pd.DataFrame(np.zeros(num_config - 1)[np.newaxis, :]) #time correlation
    deltat  = np.zeros((num_config - 1, 2), dtype = np.int)
    for n in range(num_config - 1):#time interval
        CII = orderings[:, n+1:] * orderings[:, n][:, np.newaxis]
        CII_SIt = CII.mean(axis = 0)
        cal_SIt = pd.concat([cal_SIt, pd.DataFrame(CII_SIt[np.newaxis, :])])

    cal_SIt = cal_SIt.iloc[1:]
    deltat[:, 0] = np.array(cal_SIt.columns) + 1 #time interval
    deltat[:, 1] = np.array(cal_SIt.count())     #time interval frequency

    aveS  = np.square(orderings.mean())
    aveS2 = np.square(orderings).mean()
    results[:, 0] = deltat[:, 0] * TimeStep * dt 
    results[:, 1] = (cal_SIt.mean() - aveS) / (aveS2 - aveS)
    if outputfile:
        np.savetxt(outputfile, results, fmt = '%.6f', header = names, comments = '')

    print ('------------Calculate Order time self-correlation Over---------')
    return results, names


def logOrderlife(dumpfile, orderings, ndim=3, dt=0.002, outputfile='', filetype='lammps', moltypes=''):
    """Calculate the time decay of structural ordering with 
    the initial configuration as reference

    orderings is the numpy array of structural ordering
    its shape is [atom_number, snapshot_number]
    this is the standard output format of this analysis package:
    READ: orderings = np.loadtxt(orderfile, skiprows = 1)[:, 1:]

    dumpfile is used to get the time scale
    """

    print ('------------Calculate LOG Order time self-correlation---------')
    d = readdump(dumpfile, ndim, filetype, moltypes)
    d.read_onefile()

    if d.SnapshotNumber != orderings.shape[1]:
        errorinfo = '***inconsistent number of configurations and orderings***'
        raise ValueError(errorinfo)

    results = np.zeros((d.SnapshotNumber - 1, 2))
    names = 't  St'

    results[:, 0] = (np.array(d.TimeStep[1:]) - np.array(d.TimeStep[0])) * dt 
    CII = orderings[:, 1:] * orderings[:, 0][:, np.newaxis]

    aveS  = np.square(orderings.mean())
    aveS2 = np.square(orderings).mean()
    results[:, 1] = (CII.mean(axis = 0) - aveS) / (aveS2 - aveS)

    if outputfile:
        np.savetxt(outputfile, results, fmt = '%.6f', header = names, comments = '')

    print ('------------Calculate LOG Order time self-correlation Over---------')
    return results, names


def Orderlife_FT(dumpfile, orderings, qvector, ndim=3, dt=0.002, outputfile='', filetype='lammps', moltypes='', normalization=False):
    """calculate the time decay of structural ordering in reciprocal space
        by Fourier Transformation

        orderings is the numpy array of structural ordering
        its shape is [atom_number, snapshot_number]
        this is the standard output format of this analysis package:
        READ: orderings = np.loadtxt(orderfile, skiprows=1)[:, 1:]

        dumpfile is used to get the timescale and the FFT

        qvector is the designed wavevector, such as [1, 0, 0] for 3D
    """

    print ("------Calculate Order time self-correlation by FT------")
    d = readdump(dumpfile, ndim, filetype, moltypes)
    d.read_onefile()

    print (d.SnapshotNumber)
    print (orderings.shape)

    TimeStep = d.TimeStep[1] - d.TimeStep[0]
    if TimeStep != d.TimeStep[-1] - d.TimeStep[-2]:
        print('Warning: ********time interval changes************')

    twopidl = 2.0*np.pi/d.Boxlength[0]
    qvector *= twopidl
    qvalue = np.linalg.norm(qvector)

    #calculate FT for each configuration
    sinparts = np.zeros(d.SnapshotNumber)
    cosparts = np.zeros_like(sinparts)
    for n in range(d.SnapshotNumber):
        theta = (qvector * d.Positions[n]).sum(axis=1)
        sinparts[n] = np.sum(orderings[:, n]*np.sin(theta)) #/orderings[:, n].sum()
        cosparts[n] = np.sum(orderings[:, n]*np.cos(theta)) #/orderings[:, n].sum()
        if normalization:
            sinparts[n] /= orderings[:, n].sum()
            cosparts[n] /= orderings[:, n].sum()
    
    #calculate time correlation
    results = np.zeros((d.SnapshotNumber, 2))
    counts = np.zeros(d.SnapshotNumber)
    for n in range(d.SnapshotNumber):
        for nn in range(n+1):
            results[nn, 1] += sinparts[n]*sinparts[n-nn]+cosparts[n]*cosparts[n-nn]
            counts[nn] += 1
    
    results[:, 1] /= counts
    results[:, 0] = (np.array(d.TimeStep)-d.TimeStep[0])*dt
    results[:, 1] /= results[0, 1]
    
    if outputfile:
        np.savetxt(outputfile, results, fmt='%.6f', header='t Sqt', comments='qvalue=%.3f; length=%.3f\n'%(qvalue, 2.0*np.pi/qvalue))
    
    print("------Calculate Order time self-correlation by FFT Over------")
    return results, qvalue
