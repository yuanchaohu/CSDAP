#!/usr/bin/python
# coding = utf-8
#This module is part of an analysis package

Authorinfo = """
             ------------------Name: Yuan-Chao Hu--------------
             --------------Email: ychu0213@gmail.com-----------
             ----------Web: https://yuanchaohu.github.io/------
             """

Docstr = """
        This module is used to calculate the properties of chosen particles
        The selected should be given in a numpy array of bool type
        """

import numpy as np 
import pandas as pd  
from dump import readdump

def alpha2factor(ndim):
    """ Choose factor in alpha2 calculation """

    if ndim == 3:
        return 3.0 / 5.0
    elif ndim == 2:
        return 1.0 / 2.0

def dynamics(inputfile, selection, ndim = 3, filetype = 'lammps', moltypes = '', qmax = 0, a = 0.3, dt = 0.002, ppp = [1,1,1], PBC = True, outputfile = ''):
    """
    calculate the dynamical properties of a specific group of atoms
    identified by 'selection' of bool type

    The shape of selection must be [num_of_atom, num_of_snapshot]
    """

    print ('--------Calculate Conditional Dynamics-----------')
    d = readdump(inputfile, ndim, filetype, moltypes)
    d.read_onefile()

    if d.SnapshotNumber != selection.shape[1]:
        errorinfo = '***inconsistent number of configurations and atom selection***'
        raise ValueError(errorinfo)

    TimeStep = d.TimeStep[1] - d.TimeStep[0]
    if TimeStep != (d.TimeStep[-1] - d.TimeStep[-2]):
        print ('-------Warning: dump interval changes-------')

    results = np.zeros(((d.SnapshotNumber - 1), 5))
    names  = 't  ISF  Qt  msd  alpha2'

    cal_isf  = pd.DataFrame(np.zeros((d.SnapshotNumber-1))[np.newaxis, :])
    cal_Qt   = pd.DataFrame(np.zeros((d.SnapshotNumber-1))[np.newaxis, :])
    cal_msd  = pd.DataFrame(np.zeros((d.SnapshotNumber-1))[np.newaxis, :])
    cal_alp  = pd.DataFrame(np.zeros((d.SnapshotNumber-1))[np.newaxis, :])
    deltat   = np.zeros(((d.SnapshotNumber - 1), 2), dtype = np.int) #deltat, deltatcounts
    for n in range(d.SnapshotNumber - 1): #time interval
        condition  = selection[:, n]
        RII = [i[condition] - d.Positions[n][condition] for i in d.Positions[n+1:]]
        if PBC:#remove PBC
            hmatrixinv = np.linalg.inv(d.hmatrix[n])
            for ii in range(len(RII)):
                matrixij = np.dot(RII[ii], hmatrixinv)
                RII[ii]  = np.dot(matrixij - np.rint(matrixij) * ppp, d.hmatrix[n]) 

        RII = np.array(RII) #shape [deltat, Natom_selected, ndim]
        RII_isf   = (np.cos(RII * qmax).mean(axis = 2)).mean(axis = 1) #index is timeinterval -1
        cal_isf   = pd.concat([cal_isf, pd.DataFrame(RII_isf[np.newaxis, :])])
        distance  = np.square(RII).sum(axis = 2)
        RII_Qt    = (np.sqrt(distance) <= a).mean(axis = 1)
        cal_Qt    = pd.concat([cal_Qt, pd.DataFrame(RII_Qt[np.newaxis, :])])
        cal_msd   = pd.concat([cal_msd, pd.DataFrame(distance.mean(axis = 1)[np.newaxis, :])])
        distance2 = np.square(distance).mean(axis = 1)
        cal_alp   = pd.concat([cal_alp, pd.DataFrame(distance2[np.newaxis, :])])

    cal_isf      = cal_isf.iloc[1:]
    cal_Qt       = cal_Qt.iloc[1:]
    cal_msd      = cal_msd.iloc[1:]
    cal_alp      = cal_alp.iloc[1:]
    deltat[:, 0] = np.array(cal_isf.columns) + 1 #Timeinterval
    deltat[:, 1] = np.array(cal_isf.count())     #Timeinterval frequency

    results[:, 0] = deltat[:, 0] * TimeStep * dt
    results[:, 1] = cal_isf.mean()
    results[:, 2] = cal_Qt.mean()
    results[:, 3] = cal_msd.mean()
    results[:, 4] = cal_alp.mean()
    results[:, 5] = alpha2factor(ndim) * results[:, 4] / np.square(results[:, 3]) - 1.0

    if outputfile:
        np.savetxt(outputfile, results, fmt='%.6f', header = names, comments = '')
    
    print ('--------Calculate Conditional Dynamics OVER-----------')
    return results, names

def logdynamics(inputfile, selection, ndim = 3, filetype = 'lammps', moltypes = '', qmax = 0, a = 0.3, dt = 0.002, ppp = [1,1,1], PBC = True, outputfile = ''):
    """
    calculate the dynamical properties of a specific group of atoms
    identified by 'selection' of bool type

    The shape of selection must be [num_of_atom, num_of_snapshot]
    """

    print ('--------Calculate LOG Conditional Dynamics-----------')
    d = readdump(inputfile, ndim, filetype, moltypes)
    d.read_onefile()

    if d.SnapshotNumber != selection.shape[1]:
        print (selection.shape)
        print ('--------Warning: selection must be from the initial state---------')

    results = np.zeros(((d.SnapshotNumber - 1), 5))
    names  = 't  ISF  Qt  msd  alpha2'

    results[:, 0] = (np.array(d.TimeStep[1:]) - d.TimeStep[0]) * dt

    condition = selection[:, 0]
    RII = [i[condition] - d.Positions[0][condition] for i in d.Positions[1:]]
    if PBC:#remove PBC
        hmatrixinv = np.linalg.inv(d.hmatrix[0])
        for ii in range(len(RII)):
            matrixij = np.dot(RII[ii], hmatrixinv)
            RII[ii]  = np.dot(matrixij - np.rint(matrixij) * ppp, d.hmatrix[0]) 

    RII = np.array(RII)
    results[:, 1] = (np.cos(RII * qmax).mean(axis = 2)).mean(axis = 1)
    distance      = np.square(RII).sum(axis = 2)
    results[:, 2] = (np.sqrt(distance) <= a).mean(axis = 1)
    results[:, 3] = distance.mean(axis = 1)
    distance2     = np.square(distance).mean(axis = 1)
    results[:, 4] = alpha2factor(ndim) * distance2 / np.square(results[:, 3]) - 1.0

    if outputfile:
        np.savetxt(outputfile, results, fmt = '%.6f', header = names, comments = '')

    print ('--------Calculate LOG Conditional Dynamics-----------')
    return results, names    


def Nidealfac(ndim):
    """ Choose factor of Nideal in g(r) calculation """

    if ndim == 3:
        return 4.0 / 3
    elif ndim == 2:
        return 1.0 

def partialgr(inputfile, selection, ndim = 3, filetype = 'lammps', moltypes = '', rdelta = 0.01, ppp = [1,1,1], outputfile = ''):
    """
    calculate the pair correlation function of a specific group of atoms
    identified by 'selection' of bool type

    The shape of selection must be [num_of_atom, num_of_snapshot]
    """

    print ('--------Calculate Conditional g(r)-----------')
    d = readdump(inputfile, ndim, filetype, moltypes)
    d.read_onefile()

    if d.SnapshotNumber != selection.shape[1]:
        errorinfo = '***inconsistent number of configurations and atom selection***'
        raise ValueError(errorinfo)

    MAXBIN    = int(d.Boxlength[0].min() / 2.0 / rdelta)
    grresults = np.zeros(MAXBIN)
    for n in range(d.SnapshotNumber):
        hmatrixinv = np.linalg.inv(d.hmatrix[n])
        condition  = selection[:, n]
        for i in range(d.ParticleNumber[n] - 1):
            RIJ      = d.Positions[n][i+1:] - d.Positions[n][i]
            matrixij = np.dot(RIJ, hmatrixinv)
            RIJ      = np.dot(matrixij - np.rint(matrixij) * ppp, d.hmatrix[n]) #remove PBC
            distance = np.linalg.norm(RIJ, axis = 1)
            SIJ      = condition[i+1:] * condition[i]
            Countvalue, BinEdge = np.histogram(distance, bins = MAXBIN, range = (0, MAXBIN * rdelta), weights = SIJ)
            grresults += Countvalue

    binleft    = BinEdge[:-1]   #real value of each bin edge, not index
    binright   = BinEdge[1:]   #len(Countvalue) = len(BinEdge) - 1
    Nideal     = Nidealfac(ndim) * np.pi * (binright**ndim - binleft**ndim)
    rhototal   = d.ParticleNumber[0] / np.prod(d.Boxlength[0])
    grresults  = grresults * 2 / selection.sum() / (Nideal * rhototal)

    binright = binright - 0.5 * rdelta #middle of each bin
    results  = np.column_stack((binright, grresults))
    names    = 'r  g_c(r)'
    if outputfile:
        np.savetxt(outputfile, results, fmt='%.6f', header = names, comments = '')

    print ('--------Calculate Conditional g(r) OVER-----------')
    return results, names

def partialSq(inputfile, selection, ndim = 3, filetype = 'lammps', moltypes = '', ppp = [1,1,1], qrange = 10, outputfile = ''):
    """
    calculate the structure factor of a specific group of atoms
    identified by 'selection' of bool type

    The shape of selection must be [num_of_atom, num_of_snapshot]
    """

    print ('--------Calculate Conditional S(q)-----------')
    from structurefactors import choosewavevector #(Numofq, ndim)

    d = readdump(inputfile, ndim, filetype, moltypes)
    d.read_onefile()

    if d.SnapshotNumber != selection.shape[1]:
        errorinfo = '***inconsistent number of configurations and atom selection***'
        raise ValueError(errorinfo)

    twopidl    = 2 * np.pi / d.Boxlength[0][0]
    Numofq = int(qrange / twopidl)
    wavevector = choosewavevector(Numofq, ndim)
    qvalue, qcount = np.unique(wavevector[:, 0], return_counts = True)
    sqresults = np.zeros((len(wavevector[:, 0]), 2)) #the first row accouants for wavenumber
    for n in range(d.SnapshotNumber):
        sqtotal   = np.zeros((len(wavevector[:, 0]), 2))
        condition = selection[:, n] 
        for i in range(d.ParticleNumber[n]):
            #medium   = twopidl * (d.Positions[n][i] * wavevector[:, 1:]).sum(axis = 1)
            #sqtotal += np.column_stack((np.sin(medium)*condition[i], np.cos(medium)*condition[i]))
            if condition[i]:
                medium   = twopidl * (d.Positions[n][i] * wavevector[:, 1:]).sum(axis = 1)
                sqtotal += np.column_stack((np.sin(medium), np.cos(medium)))

        sqresults[:, 1] += np.square(sqtotal).sum(axis = 1) / condition.sum() #d.ParticleNumber[n]

    sqresults[:, 0]  = wavevector[:, 0]
    sqresults[:, 1]  = sqresults[:, 1] / d.SnapshotNumber

    sqresults = pd.DataFrame(sqresults)
    results   = np.array(sqresults.groupby(sqresults[0]).mean())
    qvalue    = twopidl * np.sqrt(qvalue)
    results   = np.column_stack((qvalue, results))
    names = 'q  S(q)'

    if outputfile:
        np.savetxt(outputfile, results, fmt='%.6f', header = names, comments = '')

    print ('--------Calculate Conditional S(q) OVER-----------')
    return results, names
