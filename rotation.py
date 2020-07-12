#!/usr/bin/python
#coding = utf-8
#This module is part of an analysis package

Authorinfo = """
             ------------------Name: Yuan-Chao Hu--------------
             --------------Email: ychu0213@gmail.com-----------
             ----------Web: https://yuanchaohu.github.io/------
             """

Docstr = """
        This module is used to calculate rotational dynamics
        based on the orientation of each particle
"""

import numpy as np 
import pandas as pd 
from dumpAngular import readangular

def CRtotal(filename, ndim = 3, dt = 0.002, outputfile = ''):
    """calcualte rotational dynamics over all particles by moving average

        the time interval should be the same for average with the same deltat
    """

    print ('-------calculate overall rotational dynamics-------')
    d = readangular(filename, ndim)
    d.read_onefile()

    #-----get the unit vector-----
    velocity = [u/np.linalg.norm(u, axis = 1)[:, np.newaxis] for u in d.velocity]
    TimeStep = d.TimeStep[1] - d.TimeStep[0]
    if TimeStep != d.TimeStep[-1] - d.TimeStep[-2]:
        print ('Warning: ********time interval changes********') 
    ParticleNumber = d.ParticleNumber[0]
    if ParticleNumber != d.ParticleNumber[-1]:
        print ('Warning: ********particle number changes**********')

    results  = np.zeros((d.SnapshotNumber - 1, 2))
    names    = 't  CRt'
    cal_CRt  = pd.DataFrame(np.zeros(d.SnapshotNumber - 1)[np.newaxis, :])
    deltat   = np.zeros((d.SnapshotNumber - 1, 2), dtype = np.int)
    for n in range(d.SnapshotNumber - 1): #time interval
        CII     = velocity[n+1:] * velocity[n]
        CII_CRt = (CII.sum(axis = 2)).sum(axis = 1)
        cal_CRt = pd.concat([cal_CRt, pd.DataFrame(CII_CRt[np.newaxis, :])])

    cal_CRt = cal_CRt.iloc[1:]
    deltat[:, 0] = np.array(cal_CRt.columns) + 1 #time interval
    deltat[:, 1] = np.array(cal_CRt.count())     #time interval frequency

    results[:, 0] = deltat[:, 0] * TimeStep * dt
    results[:, 1] = cal_CRt.mean() / ParticleNumber
    if outputfile:
        np.savetxt(outputfile, results, fmt = '%.6f', header = names, comments = '')

    print ('-------calculate overall rotational dynamics over-------')
    return results, names

def logCRtotal(filename, ndim = 3, dt = 0.002, outputfile = ''):
    """calcualte rotational dynamics over all particles 

        only the initial configuration as reference
    """

    print ('-------calculate LOG overall rotational dynamics-------')
    d = readangular(filename, ndim)
    d.read_onefile()

    #-----get the unit vector-----
    velocity = [u/np.linalg.norm(u, axis = 1)[:, np.newaxis] for u in d.velocity]

    results  = np.zeros((d.SnapshotNumber - 1, 2))
    names    = 't  CRt'

    results[:, 0] = (np.array(d.TimeStep[1:]) - d.TimeStep[0]) * dt
    CII = velocity[1:] * velocity[0]
    results[:, 1] = (CII.sum(axis = 2)).mean(axis = 1)

    if outputfile:
        np.savetxt(outputfile, results, fmt = '%.6f', header = names, comments = '')

    print ('-------calculate LOG overall rotational dynamics over-------')
    return results, names

def Rorder(filename, ndim = 3, neighborfile = '', outputfile = ''):
    """rotational order parameter to characterize the structure

    local rotational symmetry over the nearest neighbors
    """

    print ('-------calculate local rotational ordering-------')
    from ParticleNeighbors import Voropp
    d = readangular(filename, ndim)
    d.read_onefile()

    #-----get the unit vector-----
    velocity = [u/np.linalg.norm(u, axis = 1)[:, np.newaxis] for u in d.velocity]

    fneighbor = open(neighborfile, 'r')
    results   = np.zeros((d.ParticleNumber[0], d.SnapshotNumber))
    for n in range(d.SnapshotNumber):
        Neighborlist = Voropp(fneighbor, d.ParticleNumber[n]) ##neighbor list [number, list....]
        for i in range(d.ParticleNumber[n]):
            CII = velocity[n][i] * velocity[n][Neighborlist[i, 1:1 + Neighborlist[i, 0]]]
            #psi = np.linalg.norm(CII, axis = 1).sum()
            #results[i, n] = psi / Neighborlist[i, 0]
            results[i, n] = CII.sum(axis = 1).mean()

    results = np.column_stack((np.arange(d.ParticleNumber[0])+1, results))
    if outputfile:
        names = 'id Psi'
        numformat = '%d ' + '%.6f ' * (results.shape[1] - 1)
        np.savetxt(outputfile, results, fmt = numformat, header = names, comments = '')

    print ('-------calculate local rotational ordering over-------')
    return results


def RorderIJ(filename, ndim = 3, UIJ = 0.9, neighborfile = '', outputfile = '', outputfileij = ''):
    """rotational order parameter to characterize the structure

    alignment of center against its neighbors by orientation
    """

    print ('-------calculate orientational alignment-------')
    from ParticleNeighbors import Voropp
    d = readangular(filename, ndim)
    d.read_onefile()

    #-----get the unit vector-----
    velocity = [u/np.linalg.norm(u, axis = 1)[:, np.newaxis] for u in d.velocity]

    fneighbor = open(neighborfile, 'r')
    results   = np.zeros((d.ParticleNumber[0], d.SnapshotNumber))
    if outputfileij: fij = open(outputfileij, 'w')
    for n in range(d.SnapshotNumber):
        Neighborlist = Voropp(fneighbor, d.ParticleNumber[n]) ##neighbor list [number, list....]
        if outputfileij: fij.write('id cn UIJ_list\n')
        for i in range(d.ParticleNumber[n]):
            CII = velocity[n][i] * velocity[n][Neighborlist[i, 1:1 + Neighborlist[i, 0]]]
            #psi = np.linalg.norm(CII, axis = 1)
            psi = CII.sum(axis = 1)
            results[i, n] = (psi > UIJ).sum()
            if outputfileij:
                fij.write('%d %d ' %(i+1, Neighborlist[i, 0]))
                for j in range(Neighborlist[i, 0]):
                    fij.write('%.6f ' %psi[j])
                fij.write('\n')

    results = np.column_stack((np.arange(d.ParticleNumber[0])+1, results))
    if outputfile:
        names = 'id UIJ'
        np.savetxt(outputfile, results, fmt = '%d', header = names, comments = '')

    if outputfileij: fij.close()
    print ('-------calculate orientational alignment over-------')
    return results