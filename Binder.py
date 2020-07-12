#!/usr/bin/python
# coding = utf-8
# This module is part of an analysis package

Authorinfo = """
             ------------------Name: Yuan-Chao Hu--------------
             --------------Email: ychu0213@gmail.com-----------
             ----------Web: https://yuanchaohu.github.io/------
             """

Docstr = """
         This module is responsible for calculating Binder Cumulant
         """

import os
import numpy  as np 
import pandas as pd 
from   dump   import readdump
from   math   import pi


class BCumulant:
    """ Compute Binder Cumulant """

    def __init__(self, inputfile, ndim, filetype = 'lammps', moltypes = '', *arg):
        """
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

        self.inputfile = inputfile
        self.ndim = ndim
        self.filetype = filetype
        self.moltypes = moltypes
        d = readdump(self.inputfile, self.ndim, self.filetype, self.moltypes)
        d.read_onefile()

        if len(d.TimeStep) > 1:
            self.TimeStep = d.TimeStep[1] - d.TimeStep[0]
            if self.TimeStep != d.TimeStep[-1] - d.TimeStep[-2]:
                print ('Warning: *********** dump interval changes **************') 
        self.ParticleNumber     = d.ParticleNumber[0] 
        if d.ParticleNumber[0] != d.ParticleNumber[-1]:
            print ('Warning: ************* Paticle Number Changes **************')
        self.ParticleType   = d.ParticleType
        self.SnapshotNumber = d.SnapshotNumber
        self.Positions = np.array(d.Positions)
        self.Boxlength = d.Boxlength[0]
        self.Boxbounds = d.Boxbounds[0]
        if not (d.Boxlength[0] == d.Boxlength[-1]).all():
            print ('Warning: *********Box Length Changed from Dump***********')
        self.typecounts = np.unique(self.ParticleType[0], return_counts = True) 
        self.Type       = self.typecounts[0]
        self.TypeNumber = self.typecounts[1]
        print ('Particle Type:', self.Type)
        print ('Particle TypeNumber:', self.TypeNumber)
        if np.sum(self.TypeNumber) != self.ParticleNumber:
            print ('Warning: ****** Sum of Indivdual Types is Not the Total Amount*******')

    def BCall(self, a = 1.0, dt = 0.002, outputfile = ''):
        """ Compute overall binder cumulant at different timescales

            Based on overlap function Qt and its corresponding dynamic susceptibility QtX4     
            a is the cutoff for the overlap function, default is 1.0 (EAM) and 0.3(LJ) (0.3<d>)
            dt is the timestep of MD simulations
            X4timeset is the peaktime scale of X4, if 0 will use the calculated one
            The Qt and X4 should be calculated first
        """
        print ('-----------------Compute Binder Cumulant Start--------------')

        results = np.zeros(((self.SnapshotNumber - 1), 4))
        names  = 't  Qt  QtX4  Binder'
        
        cal_Qt   = pd.DataFrame(np.zeros((self.SnapshotNumber-1))[np.newaxis, :])
        deltat   = np.zeros(((self.SnapshotNumber - 1), 2), dtype = np.int) #deltat, deltatcounts
        for n in range(self.SnapshotNumber - 1):  #time interval
            RII = self.Positions[n + 1:] - self.Positions[n]
            distance  = np.square(RII).sum(axis = 2)
            RII_Qt    = (np.sqrt(distance) <= a).sum(axis = 1)
            cal_Qt    = pd.concat([cal_Qt, pd.DataFrame(RII_Qt[np.newaxis, :])])
        
        cal_Qt       = cal_Qt.iloc[1:]
        deltat[:, 0] = np.array(cal_Qt.columns) + 1 #Timeinterval
        deltat[:, 1] = np.array(cal_Qt.count())     #Timeinterval frequency

        results[:, 0] = deltat[:, 0] * self.TimeStep * dt 
        results[:, 1] = cal_Qt.mean() / self.ParticleNumber
        results[:, 2] = ((cal_Qt**2).mean() - (cal_Qt.mean())**2) / self.ParticleNumber

        binderunit = (cal_Qt - cal_Qt.mean())**2
        results[:, 3] = (1 / 3.0) * (binderunit**2).mean() / (binderunit.mean())**2 -1
        if outputfile:
            np.savetxt(outputfile, results, fmt='%.6f', header = names, comments = '')

        print ('-----------------Compute Binder Cumulant Done--------------')
        return results, names

    def BCtype(self, a = 1.0, dt = 0.002, outputfile = ''):
        """ Compute partial binder cumulant based on different particle types at different timescales

            Based on overlap function Qt and its corresponding dynamic susceptibility QtX4     
            a is the cutoff for the overlap function, default is 1.0 (EAM) and 0.3(LJ) (0.3<d>)
            dt is the timestep of MD simulations
            X4timeset is the peaktime scale of X4, if 0 will use the calculated one
            The Qt and X4 should be calculated first
        """
        print ('-----------------Compute Partial Binder Cumulant Start--------------')

        partialresults = [] #a list containing results of all particle types
            
        for i in self.Type:  #loop over different particle types
            TYPESET = np.where(np.array(self.ParticleType) == i, 1, 0)

            results = np.zeros(((self.SnapshotNumber - 1), 4))
            names  = 't  Qt  QtX4  Binder'
            
            cal_Qt   = pd.DataFrame(np.zeros((self.SnapshotNumber-1))[np.newaxis, :])
            deltat   = np.zeros(((self.SnapshotNumber - 1), 2), dtype = np.int) #deltat, deltatcounts
            for n in range(self.SnapshotNumber - 1):  #time interval
                RII = self.Positions[n + 1:] - self.Positions[n]
                distance  = np.square(RII).sum(axis = 2)
                RII_Qt    = ((np.sqrt(distance) <= a) * TYPESET[n + 1:]).sum(axis = 1)
                cal_Qt    = pd.concat([cal_Qt, pd.DataFrame(RII_Qt[np.newaxis, :])])

            cal_Qt       = cal_Qt.iloc[1:]
            deltat[:, 0] = np.array(cal_Qt.columns) + 1 #Timeinterval
            deltat[:, 1] = np.array(cal_Qt.count())     #Timeinterval frequency

            results[:, 0] = deltat[:, 0] * self.TimeStep * dt 
            results[:, 1] = cal_Qt.mean() / self.ParticleNumber
            results[:, 2] = ((cal_Qt**2).mean() - (cal_Qt.mean())**2) / self.ParticleNumber

            binderunit = (cal_Qt - cal_Qt.mean())**2
            results[:, 3] = (1 / 3.0) * (binderunit**2).mean() / (binderunit.mean())**2 -1
            if outputfile:
                np.savetxt('Type' + str(i) + '.' + outputfile, results, fmt='%.6f', header = names, comments = '')

            partialresults.append(results)
        
        print ('-----------------Compute Partial Binder Cumulant Done--------------')
        return partialresults, names