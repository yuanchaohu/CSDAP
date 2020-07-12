#!/usr/bin/python
# coding = utf-8
#This module is part of an analysis package

Authorinfo = """
             ------------------Name: Yuan-Chao Hu--------------
             --------------Email: ychu0213@gmail.com-----------
             ----------Web: https://yuanchaohu.github.io/------
             """

Docstr = """
         This module is responsible for calculating particle level dynamics
         Compute self-intermediate scattering functions ISF, dynamic susceptibility ISFX4 based on ISF
         Overlap function Qt and its corresponding dynamic susceptibility QtX4
         Mean-square displacements msd; non-Gaussion parameter alpha2
         four-point dynamic structure factor of fast and slow particles, respectively

         The module also computes corresponding particle type related dynamics by using the function partial()
         The module accounts for systems ranging from unary to senary
         """

import os
import numpy  as np 
import pandas as pd 
from dump import readdump
from math import pi
from structurefactors import wavevector3d, wavevector2d 

def alpha2factor(ndim):
    """ Choose factor in alpha2 calculation """

    if ndim == 3:
        return 3.0 / 5.0
    elif ndim == 2:
        return 1.0 / 2.0

def choosewavevector(Numofq, ndim):
    """ Choose Wavevector in dynamic structure factor """

    if ndim == 3:
        return wavevector3d(Numofq)
    elif ndim == 2:
        return wavevector2d(Numofq)

class dynamics:
    """ Compute particle-level dynamics """

    def __init__(self, inputfile, ndim, filetype = 'lammps', moltypes = '', ppp = [1,1,1], PBC = True,  *arg):
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
        self.ppp = ppp
        self.PBC = PBC
        d = readdump(self.inputfile, self.ndim, self. filetype, self.moltypes)
        d.read_onefile()

        if len(d.TimeStep) > 1:
            self.TimeStep = d.TimeStep[1] - d.TimeStep[0]
            if self.TimeStep != d.TimeStep[-1] - d.TimeStep[-2]:
                print ('Warning: *********** dump interval changes **************') 
        self.ParticleNumber     = d.ParticleNumber[0] 
        if d.ParticleNumber[0] != d.ParticleNumber[-1]:
            print ('Warning: ************* Paticle Number Changes **************')
        self.ParticleType   = d.ParticleType
        self.Positions      = d.Positions
        self.SnapshotNumber = d.SnapshotNumber
        self.Boxlength      = d.Boxlength[0]
        if not (d.Boxlength[0] == d.Boxlength[-1]).all():
            print ('Warning: *********Box Length Changed from Dump***********')
        self.Boxbounds  = d.Boxbounds[0]
        self.typecounts = np.unique(self.ParticleType[0], return_counts = True) 
        self.Type       = self.typecounts[0].astype(np.int32)
        self.TypeNumber = self.typecounts[1]
        print ('Particle Type:', self.Type)
        print ('Particle TypeNumber:', self.TypeNumber)
        if np.sum(self.TypeNumber) != self.ParticleNumber:
            print ('Warning: ****** Sum of Indivdual Types is Not the Total Amount*******')

        self.hmatrix = d.hmatrix


    def total(self, qmax, a = 1.0, dt = 0.002, outputfile = ''):
        """ Compute self-intermediate scattering functions ISF, dynamic susceptibility ISFX4 based on ISF
            Overlap function Qt and its corresponding dynamic susceptibility QtX4
            Mean-square displacements msd; non-Gaussion parameter alpha2
        
            qmax is the wavenumber corresponding to the first peak of structure factor
            a is the cutoff for the overlap function, default is 1.0 (EAM) and 0.3(LJ) (0.3<d>)
            dt is the timestep of MD simulations
            ppp is periodic boundary conditions
        """
        print ('-----------------Compute Overall Dynamics--------------------')

        results = np.zeros(((self.SnapshotNumber - 1), 7))
        names  = 't  ISF  ISFX4  Qt  QtX4  msd  alpha2'
        
        cal_isf  = pd.DataFrame(np.zeros((self.SnapshotNumber-1))[np.newaxis, :])
        cal_Qt   = pd.DataFrame(np.zeros((self.SnapshotNumber-1))[np.newaxis, :])
        cal_msd  = pd.DataFrame(np.zeros((self.SnapshotNumber-1))[np.newaxis, :])
        cal_alp  = pd.DataFrame(np.zeros((self.SnapshotNumber-1))[np.newaxis, :])
        deltat   = np.zeros(((self.SnapshotNumber - 1), 2), dtype = np.int) #deltat, deltatcounts
        for n in range(self.SnapshotNumber - 1):  #time interval
            RII = self.Positions[n + 1:] - self.Positions[n]
            if self.PBC:#remove PBC if True
                hmatrixinv = np.linalg.inv(self.hmatrix[n])
                for ii in range(len(RII)):
                    matrixij = np.dot(RII[ii], hmatrixinv)
                    RII[ii]  = np.dot(matrixij - np.rint(matrixij) * self.ppp, self.hmatrix[n]) #remove PBC

            RII_isf   = (np.cos(RII * qmax).mean(axis = 2)).sum(axis = 1) #index is timeinterval -1
            cal_isf   = pd.concat([cal_isf, pd.DataFrame(RII_isf[np.newaxis, :])])
            distance  = np.square(RII).sum(axis = 2)
            RII_Qt    = (np.sqrt(distance) <= a).sum(axis = 1)
            cal_Qt    = pd.concat([cal_Qt, pd.DataFrame(RII_Qt[np.newaxis, :])])
            cal_msd   = pd.concat([cal_msd, pd.DataFrame(distance.sum(axis = 1)[np.newaxis, :])])
            distance2 = np.square(distance).sum(axis = 1)
            cal_alp   = pd.concat([cal_alp, pd.DataFrame(distance2[np.newaxis, :])])
       
        cal_isf      = cal_isf.iloc[1:]
        cal_Qt       = cal_Qt.iloc[1:]
        cal_msd      = cal_msd.iloc[1:]
        cal_alp      = cal_alp.iloc[1:]
        deltat[:, 0] = np.array(cal_isf.columns) + 1 #Timeinterval
        deltat[:, 1] = np.array(cal_isf.count())     #Timeinterval frequency

        results[:, 0] = deltat[:, 0] * self.TimeStep * dt 
        results[:, 1] = cal_isf.mean() / self.ParticleNumber
        results[:, 2] = ((cal_isf**2).mean() - (cal_isf.mean())**2) / self.ParticleNumber
        results[:, 3] = cal_Qt.mean() / self.ParticleNumber
        results[:, 4] = ((cal_Qt**2).mean() - (cal_Qt.mean())**2) / self.ParticleNumber
        results[:, 5] = cal_msd.mean() / self.ParticleNumber
        results[:, 6] = cal_alp.mean() / self.ParticleNumber
        results[:, 6] = alpha2factor(self.ndim) * results[:, 6] / np.square(results[:, 5]) - 1.0

        if outputfile:
            np.savetxt(outputfile, results, fmt='%.6f', header = names, comments = '')
        
        print ('-----------------Compute Overall Dynamics Over--------------------')
        return results, names

    def partial(self, qmax, a = 1.0, dt = 0.002, outputfile = ''):
        """ Compute self-intermediate scattering functions ISF, dynamic susceptibility ISFX4 based on ISF
            Overlap function Qt and its corresponding dynamic susceptibility QtX4
            Mean-square displacements msd; non-Gaussion parameter alpha2
        
            qmax is the wavenumber corresponding to the first peak of structure factor
            qmax accounts for six components so it is a list
            a is the cutoff for the overlap function, default is 1.0 (EAM) and 0.3(LJ) (0.3<d>)
            dt is the timestep of MD simulations
            ppp is periodic boundary conditions
        """
        print ('-----------------Compute Partial Dynamics--------------------')

        partialresults = [] #a list containing results of all particle types           
        for i in self.Type:  #loop over different particle types
            #TYPESET = np.where(np.array(self.ParticleType) == i, 1, 0).astype(np.int)
            TYPESET = [j == i for j in self.ParticleType]

            results = np.zeros(((self.SnapshotNumber - 1), 7))
            names  = 't  ISF  ISFX4  Qt  QtX4  msd  alpha2'
            
            cal_isf  = pd.DataFrame(np.zeros((self.SnapshotNumber-1))[np.newaxis, :])
            cal_Qt   = pd.DataFrame(np.zeros((self.SnapshotNumber-1))[np.newaxis, :])
            cal_msd  = pd.DataFrame(np.zeros((self.SnapshotNumber-1))[np.newaxis, :])
            cal_alp  = pd.DataFrame(np.zeros((self.SnapshotNumber-1))[np.newaxis, :])
            deltat   = np.zeros(((self.SnapshotNumber - 1), 2), dtype = np.int) #deltat, deltatcounts
            for n in range(self.SnapshotNumber - 1):  #loop over time intervals
                #RII    = self.Positions[n + 1:] - self.Positions[n]
                RII  = [ii[TYPESET[n]] - self.Positions[n][TYPESET[n]] for ii in self.Positions[n+1:]]
                if self.PBC:
                    hmatrixinv = np.linalg.inv(self.hmatrix[n])
                    for ii in range(len(RII)):
                        matrixij = np.dot(RII[ii], hmatrixinv)
                        RII[ii]  = np.dot(matrixij - np.rint(matrixij) * self.ppp, self.hmatrix[n]) #remove PBC

                RII       = np.array(RII)
                RII_isf   = ((np.cos(RII * qmax[i - 1]).mean(axis = 2))).sum(axis = 1) #index is timeinterval -1
                cal_isf   = pd.concat([cal_isf, pd.DataFrame(RII_isf[np.newaxis, :])])
                distance  = np.square(RII).sum(axis = 2)
                RII_Qt    = ((np.sqrt(distance) <= a)).sum(axis = 1)
                cal_Qt    = pd.concat([cal_Qt, pd.DataFrame(RII_Qt[np.newaxis, :])])
                cal_msd   = pd.concat([cal_msd, pd.DataFrame(distance.sum(axis = 1)[np.newaxis, :])])
                distance2 = (np.square(distance)).sum(axis = 1)
                cal_alp   = pd.concat([cal_alp, pd.DataFrame(distance2[np.newaxis, :])])
        
            cal_isf      = cal_isf.iloc[1:]
            cal_Qt       = cal_Qt.iloc[1:]
            cal_msd      = cal_msd.iloc[1:]
            cal_alp      = cal_alp.iloc[1:]
            deltat[:, 0] = np.array(cal_isf.columns) + 1 #Timeinterval
            deltat[:, 1] = np.array(cal_isf.count())     #Timeinterval frequency

            results[:, 0] = deltat[:, 0] * self.TimeStep * dt 
            results[:, 1] = cal_isf.mean() / self.TypeNumber[i - 1]
            results[:, 2] = ((cal_isf**2).mean() - (cal_isf.mean())**2) / self.TypeNumber[i - 1]
            results[:, 3] = cal_Qt.mean() / self.TypeNumber[i - 1]
            results[:, 4] = ((cal_Qt**2).mean() - (cal_Qt.mean())**2) / self.TypeNumber[i - 1]
            results[:, 5] = cal_msd.mean() / self.TypeNumber[i - 1]
            results[:, 6] = cal_alp.mean() / self.TypeNumber[i - 1]
            results[:, 6] = alpha2factor(self.ndim) * results[:, 6] / np.square(results[:, 5]) - 1.0

            if outputfile:
                np.savetxt('Type' + str(i) + '.' + outputfile, results, fmt='%.6f', header = names, comments = '')
            
            partialresults.append(results)

        print ('-----------------Compute Partial Dynamics Over--------------------')
        return partialresults, names

    def slowS4(self, X4time, dt = 0.002, a = 1.0, qrange = 10, outputfile = ''):
        """ Compute four-point dynamic structure factor at peak timescale of dynamic susceptibility

            Based on overlap function Qt and its corresponding dynamic susceptibility QtX4     
            a is the cutoff for the overlap function, default is 1.0 (EAM) and 0.3(LJ) (0.3<d>)
            X4time is the peaktime scale of X4
            dt is the timestep in MD simulations
            Dynamics should be calculated before computing S4
            Only considered the particles which are slow
            ppp is the periodic boundary conditions
        """
        print ('-----------------Compute dynamic S4(q) of slow particles --------------')

        X4time = int(X4time / dt / self.TimeStep)
        twopidl = 2 * pi / self.Boxlength[0]
        Numofq = int(qrange / twopidl)

        wavevector = choosewavevector(Numofq, self.ndim) #Only S4(q) at low wavenumber range is interested
        qvalue, qcount = np.unique(wavevector[:, 0], return_counts = True)
        sqresults = np.zeros((len(wavevector[:, 0]), 3)) #the first row accouants for wavenumber

        for n in range(self.SnapshotNumber - X4time):
            RII = self.Positions[n + X4time] - self.Positions[n]
            if self.PBC:
                hmatrixinv = np.linalg.inv(self.hmatrix[n])
                matrixij   = np.dot(RII, hmatrixinv)
                RII        = np.dot(matrixij - np.rint(matrixij) * self.ppp, self.hmatrix[n]) #remove PBC

            #RII = np.square(RII).sum(axis = 1)
            #RII = np.where(np.sqrt(RII) <= a, 1, 0)
            RII = np.linalg.norm(RII, axis = 1)
            RII = np.where(RII <= a, 1, 0)
            sqtotal = np.zeros((len(wavevector[:, 0]), 2))
            for i in range(self.ParticleNumber):
                medium   = twopidl * (self.Positions[n][i] * wavevector[:, 1:]).sum(axis = 1)
                sqtotal[:, 0] += np.sin(medium) * RII[i]
                sqtotal[:, 1] += np.cos(medium) * RII[i]
            
            sqresults[:, 1] += np.square(sqtotal).sum(axis = 1) / self.ParticleNumber
            sqresults[:, 2] += sqtotal[:, 1]

        sqresults[:, 0]  = wavevector[:, 0]
        sqresults[:, 1]  = sqresults[:, 1] / (self.SnapshotNumber - X4time)
        sqresults[:, 2]  = np.square(sqresults[:, 2] / (self.SnapshotNumber - X4time)) / self.ParticleNumber

        sqresults = pd.DataFrame(sqresults)
        results   = np.array(sqresults.groupby(sqresults[0]).mean())
        results[:, 1] = results[:, 0] - results[:, 1] / qcount

        qvalue    = twopidl * np.sqrt(qvalue)
        results   = np.column_stack((qvalue, results))
        names = 'q  S4a(q)  S4b(q)'
        if outputfile:
            np.savetxt(outputfile, results, fmt='%.6f', header = names, comments = '')

        print ('--------- Compute S4(q) of slow particles over ------')
        return results, names

    def fastS4(self, a = 1.0, dt = 0.002, X4timeset = 0, qrange = 10, outputfile = ''):
        """ Compute four-point dynamic structure factor at peak timescale of dynamic susceptibility

            Based on overlap function Qt and its corresponding dynamic susceptibility QtX4     
            a is the cutoff for the overlap function, default is 1.0 (EAM) and 0.3(LJ) (0.3<d>)
            dt is the timestep of MD simulations
            X4timeset is the peaktime scale of X4, if 0 will use the calculated one
            Dynamics should be calculated before computing S4
            Only considered the particles which are fast
            The Qt and X4 should be calculated first
            ppp is the periodic boundary conditions
        """
        print ('-----------------Compute dynamic S4(q) of fast particles --------------')

        #-----------calculte overall dynamics first----------------
        results = np.zeros(((self.SnapshotNumber - 1), 3))
        names  = 't  Qt  QtX4'
        
        cal_Qt   = pd.DataFrame(np.zeros((self.SnapshotNumber-1))[np.newaxis, :])
        deltat   = np.zeros(((self.SnapshotNumber - 1), 2), dtype = np.int) #deltat, deltatcounts
        for n in range(self.SnapshotNumber - 1):  #time interval
            RII = self.Positions[n + 1:] - self.Positions[n]
            if self.PBC:
                hmatrixinv = np.linalg.inv(self.hmatrix[n])
                for ii in range(len(RII)):
                    matrixij = np.dot(RII[ii], hmatrixinv)
                    RII[ii]  = np.dot(matrixij - np.rint(matrixij) * self.ppp, self.hmatrix[n]) #remove PBC

            distance  = np.square(RII).sum(axis = 2)
            RII_Qt    = (np.sqrt(distance) >= a).sum(axis = 1)
            cal_Qt    = pd.concat([cal_Qt, pd.DataFrame(RII_Qt[np.newaxis, :])])
        
        cal_Qt       = cal_Qt.iloc[1:]
        deltat[:, 0] = np.array(cal_Qt.columns) + 1 #Timeinterval
        deltat[:, 1] = np.array(cal_Qt.count())     #Timeinterval frequency

        results[:, 0] = deltat[:, 0] * self.TimeStep * dt 
        results[:, 1] = cal_Qt.mean() / self.ParticleNumber
        results[:, 2] = ((cal_Qt**2).mean() - (cal_Qt.mean())**2) / self.ParticleNumber
        if outputfile:
            np.savetxt('Dynamics.' + outputfile, results, fmt='%.6f', header = names, comments = '')

        #-----------calculte S4(q) of fast particles----------------
        twopidl = 2 * pi / self.Boxlength[0]
        Numofq = int(qrange / twopidl)

        wavevector = choosewavevector(Numofq, self.ndim) #Only S4(q) at low wavenumber range is interested
        qvalue, qcount = np.unique(wavevector[:, 0], return_counts = True)
        sqresults = np.zeros((len(wavevector[:, 0]), 3)) #the first row accouants for wavenumber

        if X4timeset:
            X4time = int(X4timeset / dt / self.TimeStep)
        else:
            X4time = deltat[results[:, 2].argmax(), 0] 

        for n in range(self.SnapshotNumber - X4time):
            RII = self.Positions[n + X4time] - self.Positions[n]
            if self.PBC:
                hmatrixinv = np.linalg.inv(self.hmatrix[n])
                matrixij   = np.dot(RII, hmatrixinv)
                RII        = np.dot(matrixij - np.rint(matrixij) * self.ppp, self.hmatrix[n]) #remove PBC

            #RII = np.square(RII).sum(axis = 1)
            #RII = np.where(np.sqrt(RII) >= a, 1, 0)
            RII = np.linalg.norm(RII, axis = 1)
            RII = np.where(RII >= a, 1, 0)
            sqtotal = np.zeros((len(wavevector[:, 0]), 2))
            for i in range(self.ParticleNumber):
                medium   = twopidl * (self.Positions[n][i] * wavevector[:, 1:]).sum(axis = 1)
                sqtotal[:, 0] += np.sin(medium) * RII[i]
                sqtotal[:, 1] += np.cos(medium) * RII[i]
            
            sqresults[:, 1] += np.square(sqtotal).sum(axis = 1) / self.ParticleNumber
            sqresults[:, 2] += sqtotal[:, 1]

        sqresults[:, 0]  = wavevector[:, 0]
        sqresults[:, 1]  = sqresults[:, 1] / (self.SnapshotNumber - X4time)
        sqresults[:, 2]  = np.square(sqresults[:, 2] / (self.SnapshotNumber - X4time)) / self.ParticleNumber

        sqresults = pd.DataFrame(sqresults)
        results   = np.array(sqresults.groupby(sqresults[0]).mean())
        results[:, 1] = results[:, 0] - results[:, 1] / qcount

        qvalue    = twopidl * np.sqrt(qvalue)
        results   = np.column_stack((qvalue, results))
        names = 'q  S4a(q)  S4b(q)'
        if outputfile:
            np.savetxt(outputfile, results, fmt='%.6f', header = names, comments = '')

        print ('--------- Compute S4(q) of fast particles over ------')
        return results, names