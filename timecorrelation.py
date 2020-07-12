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

def Orderlife(dumpfile, orderings, ndim = 3, dt = 0.002, outputfile = ''):
    """Calculate the time decay of structural ordering

    orders is the numpy array of structural ordering
    its shape is [atom_number, snapshot_number]
    this is the standard output format of this analysis package:
    READ: orders = np.loadtxt(orderfile, skiprows = 1)[:, 1:]

    dumpfile is used to get the time scale
    """

    print ('------------Calculate Order time self-correlation---------')
    d = readdump(dumpfile, ndim)
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

def logOrderlife(dumpfile, orderings, ndim = 3, dt = 0.002, outputfile = ''):
    """Calculate the time decay of structural ordering with 
    the initial configuration as reference

    orders is the numpy array of structural ordering
    its shape is [atom_number, snapshot_number]
    this is the standard output format of this analysis package:
    READ: orders = np.loadtxt(orderfile, skiprows = 1)[:, 1:]

    dumpfile is used to get the time scale
    """

    print ('------------Calculate LOG Order time self-correlation---------')
    d = readdump(dumpfile, ndim)
    d.read_onefile()

    if d.SnapshotNumber != orderings.shape[1]:
        errorinfo = '***inconsistent number of configurations and orderings***'
        raise ValueError

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