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

    results  = np.zeros((d.SnapshotNumber - 1, 3))
    names    = 't  CRt X4'
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
    results[:, 2] = ((cal_CRt**2).mean()-(cal_CRt.mean())**2) / ParticleNumber
    if outputfile:
        np.savetxt(outputfile, results, fmt = '%.6f', header = names, comments = '')

    print ('-------calculate overall rotational dynamics over-------')
    return results, names

def S4(file_positions, file_orientations, ndim, X4time, filetype='lammps', moltypes='', dt=0.002, phi=0.2, qrange=10, outputfile=''):
    """ Compute four-point dynamic structure factor at peak timescale of dynamic susceptibility

        Based on dynamics overlap function CRtotal and its corresponding dynamic susceptibility X4     
        file_positions: atomic positions
        file_orientations: atomic orientations
        phi is the cutoff for the dynamics overlap function
        X4time is the peaktime scale of X4
        dt is the timestep in MD simulations
        Dynamics should be calculated before computing S4
        Only considered the particles which are rotationally slow
    """
    
    print ('-----Compute dynamic S4(q) of rotational slow particles-----')
    from WaveVector import choosewavevector
    from dump import readdump
    from math import pi

    #read positions
    d1 = readdump(file_positions, ndim, filetype, moltypes)
    d1.read_onefile()

    #read orientations
    d2 = readangular(file_orientations, ndim)
    d2.read_onefile()
    #get unit vector
    velocity = [u/np.linalg.norm(u, axis=1)[:, np.newaxis] for u in d2.velocity]

    #check files
    if d1.SnapshotNumber != d2.SnapshotNumber:
        print ('warning: ******check configurations*****')
    if d1.ParticleNumber[0] != d2.ParticleNumber[0]:
        print('warning: ******check configurations*****')
    if d1.TimeStep[0] != d2.TimeStep[0]:
        print('warning: ******check configurations*****')
    TimeStep = d1.TimeStep[1] - d1.TimeStep[0]
    if TimeStep != d1.TimeStep[-1] - d1.TimeStep[-2]:
        print('Warning: *****time interval changes*****')
    ParticleNumber = d1.ParticleNumber[0]
    if ParticleNumber != d1.ParticleNumber[-1]:
        print('Warning: *****particle number changes*****')

    #calculate dynamics and structure factor
    X4time = int(X4time / dt / TimeStep)
    twopidl = 2 * pi / d1.Boxlength[0][0]
    Numofq = int(qrange / twopidl)

    wavevector = choosewavevector(Numofq, ndim) #Only S4(q) at low wavenumber range is interested
    qvalue, qcount = np.unique(wavevector[:, 0], return_counts = True)
    sqresults = np.zeros((wavevector.shape[0], 2)) #the first row accouants for wavenumber

    for n in range(d1.SnapshotNumber - X4time):
        RII = (velocity[n+X4time] * velocity[n]).sum(axis=1)
        RII = np.where(RII >= phi, 1, 0)
        
        sqtotal = np.zeros((wavevector.shape[0], 2))
        for i in range(ParticleNumber):
            medium = twopidl * (d1.Positions[n][i] * wavevector[:, 1:]).sum(axis = 1)
            sqtotal[:, 0] += np.sin(medium) * RII[i]
            sqtotal[:, 1] += np.cos(medium) * RII[i]
        
        sqresults[:, 1] += np.square(sqtotal).sum(axis = 1) / ParticleNumber
        
    sqresults[:, 0]  = wavevector[:, 0]
    sqresults[:, 1] /= (d1.SnapshotNumber - X4time)
    
    sqresults = pd.DataFrame(sqresults)
    results   = np.array(sqresults.groupby(sqresults[0]).mean())
    
    qvalue    = twopidl * np.sqrt(qvalue)
    results   = np.column_stack((qvalue, results))
    names = 'q  S4'
    if outputfile:
        np.savetxt(outputfile, results, fmt='%.6f', header = names, comments = '')

    print ('--------- Compute S4(q) of slow particles over ------')
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
            results[i, n] = np.abs(CII.sum(axis = 1)).mean()

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
            psi = np.abs(CII.sum(axis = 1))
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
