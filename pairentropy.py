#!/usr/bin/python
# coding = utf-8
#This module is part of an analysis package

Authorinfo = """
             ----------Name: Qi Liu [lq4866@gmail.com]--------------------
             ----------Name: Yuan-Chao Hu [ychu0213@gmail.com]------------
             ----------Web: https://yuanchaohu.github.io/-----------------
             """

Docstr = """
         This module is responsible for calculating pair entropy S2
         The code is written refering to 'Allen book P183' covering unary to Quinary systems

         The code accounts for both orthogonal and triclinic cells
         """

import os
import numpy as np 
from   dump  import readdump

def Nidealfac(ndim):
    """ Choose factor of Nideal in g(r) calculation """

    if ndim == 3:
        return 4.0 / 3
    elif ndim == 2:
        return 1.0

def Areafac(ndim):
    """ Choose factor of area in S2 calculation """

    if ndim == 3:
        return 4.0  #4 * PI * R2
    elif ndim == 2:
        return 2.0  #2 * PI * R

class S2:
    """ Compute pair entropy S2 and then Average S2 over different Snapshots """

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
            self.TimeStep     = d.TimeStep[1] - d.TimeStep[0]
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
        self.hmatrix    = d.hmatrix
        self.Volume     = np.prod(self.Boxlength)
        self.rhototal   = self.ParticleNumber / self.Volume #number density of the total system
        self.typecounts = np.unique(self.ParticleType[0], return_counts = True) 
        self.Type       = self.typecounts[0]
        self.TypeNumber = self.typecounts[1]
        self.rhotype    = self.TypeNumber / self.Volume
        print ('Particle Type:', self.Type)
        print ('Particle TypeNumber:', self.TypeNumber)
        if np.sum(self.TypeNumber) != self.ParticleNumber:
            print ('Warning: ****** Sum of Indivdual Types is Not the Total Amount*******')

    def timeave(self, ppp, avetime, rdelta = 0.01, dt = 0.002, outputfile = ''):
        """ Calculate Time Averaged S2
            
            outputfile is file name storing outputs
            ppp is periodic boundary conditions, set 1 for yes and 0 for no, should be a list
            avetime is the time scale used to average S2, in time unit; set 0 to get results of individual snapshot
            rdelta is the bin size of gr, default is 0.01
            dt is timestep of a simulation, default is 0.002
        """

        if len(self.Type) == 1:
            S2results      = self.Unary(ppp, rdelta)
        if len(self.Type) == 2: 
            S2results      = self.Binary(ppp, rdelta)
        if len(self.Type) == 3: 
            S2results =    self.Ternary(ppp, rdelta)
        if len(self.Type) == 4: 
            S2results      = self.Quarternary(ppp, rdelta)
        if len(self.Type) == 5: 
            S2results      = self.Quinary(ppp, rdelta)

        if avetime:
            avetime    = int(avetime / dt / self.TimeStep)
            S2average  = np.zeros((self.ParticleNumber, self.SnapshotNumber - avetime)) 
            for n in range(self.SnapshotNumber - avetime):
                S2average[:, n] = S2results[:, n: n + avetime + 1].mean(axis = 1)
            results = np.column_stack((np.arange(self.ParticleNumber) + 1, S2average))
        else:
            results = np.column_stack((np.arange(self.ParticleNumber) + 1, S2results))

        names      = 'id   S2_of_each_snapshot'
        fileformat = '%d ' + '%.6f ' * (self.SnapshotNumber - avetime)

        if outputfile:
            np.savetxt(outputfile, results, fmt = fileformat, header = names, comments = '')
        
        print ('---------- Get S2 results over ---------')
        return results, names

    def spatialcorr(self, outputs2, ppp, avetime, rdelta = 0.01, dt = 0.002, outputcorr = ''):
        """ Calculate Spatial Correlation of Time Averaged S2
            
            Excuting this function will excute the function timeave() first, so the averaged S2 will be output
            outputcorr is file name storing outputs of spatial correlation of time averaged S2
            outputs2 is file name storing outputs of time averaged S2
            ppp is periodic boundary conditions, set 1 for yes and 0 for no, should be a list
            avetime is the time scale used to average S2, in time unit; set 0 to get results of individual snapshot
            rdelta is the bin size of gr, default is 0.01
            dt is timestep of a simulation, default is 0.002
        """
        print ('--------- Calculating Spatial Correlation of S2 ------')

        timeaves2      = self.timeave(outputs2, ppp, avetime, rdelta, dt, results_path)[:, 1:].T 
        MAXBIN         = int(self.Boxlength.min() / 2.0 / rdelta)
        grresults      = np.zeros((MAXBIN, 3))
        SnapshotNumber = len(timeaves2[:, 0])
        Positions      = self.Positions[:SnapshotNumber]
        for n in range(SnapshotNumber):
            hmatrixinv = np.linalg.inv(self.hmatrix[n])
            for i in range(self.ParticleNumber - 1):
                RIJ      = Positions[n][i+1:] - Positions[n][i]
                #periodic = np.where(np.abs(RIJ / self.Boxlength[np.newaxis, :]) > 0.50, np.sign(RIJ), 0).astype(np.int)
                #RIJ -= self.Boxlength * periodic * ppp    #remove PBC
                matrixij = np.dot(RIJ, hmatrixinv)
                RIJ      = np.dot(matrixij - np.rint(matrixij) * ppp, self.hmatrix[n]) #remove PBC
                distance = np.sqrt(np.square(RIJ).sum(axis = 1))
                Countvalue, BinEdge = np.histogram(distance, bins = MAXBIN, range = (0, MAXBIN * rdelta))
                grresults[:, 0]    += Countvalue
                S2IJ                = timeaves2[n, i + 1:] * timeaves2[n, i]
                Countvalue, BinEdge = np.histogram(distance, bins = MAXBIN, range = (0, MAXBIN * rdelta), weights = S2IJ)
                grresults[:, 1]    += Countvalue

        binleft          = BinEdge[:-1]   #real value of each bin edge, not index
        binright         = BinEdge[1:]    #len(Countvalue) = len(BinEdge) - 1
        Nideal           = Nidealfac(self.ndim) * np.pi * (binright**self.ndim - binleft**self.ndim) 
        grresults       *= 2 / self.ParticleNumber / SnapshotNumber / (Nideal[:, np.newaxis] * self.rhototal)
        grresults[:, 2]  = np.where(grresults[:, 0] != 0, grresults[:, 1] / grresults[:, 0], np.nan)
        binright         = binright - 0.5 * rdelta #middle of each bin
        results          = np.column_stack((binright, grresults))
        names            = 'r   g(r)   gs2(r)   gs2/gr'
        if outputcorr:
            np.savetxt(outputcorr, results, fmt='%.6f', header = names, comments = '')
        
        print ('---------- Get gs2(r) results over ---------')
        return results, names

    def Unary(self, ppp, rdelta):
        print ('--------- This is a Unary System ------')

        MAXBIN      = int(self.Boxlength.min() / 2.0 / rdelta)
        S2results   = np.zeros((self.ParticleNumber, self.SnapshotNumber))
        for n in range(self.SnapshotNumber):
            hmatrixinv = np.linalg.inv(self.hmatrix[n])
            for i in range(self.ParticleNumber):
                RIJ          = np.delete(self.Positions[n], i, axis = 0) - self.Positions[n][i]
                matrixij     = np.dot(RIJ, hmatrixinv)
                RIJ          = np.dot(matrixij - np.rint(matrixij) * ppp, self.hmatrix[n]) #remove periodic boundary conditions
                distance     = np.sqrt(np.square(RIJ).sum(axis = 1))
                
                particlegr          = np.zeros(MAXBIN) #11
                Countvalue, BinEdge = np.histogram(distance, bins = MAXBIN, range = (0, MAXBIN * rdelta)) #1-1; 
                binleft             = BinEdge[:-1]          #real value of each bin edge, not index
                binright            = BinEdge[1:]           #len(Countvalue) = len(BinEdge) - 1
                Nideal              = Nidealfac(self.ndim) * np.pi * (binright**self.ndim - binleft**self.ndim)
                particlegr          = Countvalue / Nideal / self.rhototal
                integralgr          = (particlegr * np.log(particlegr + 1e-12) - (particlegr - 1)) * self.rhototal
                binright           -= 0.5 * rdelta                                  #middle of each bin
                S2results[i, n]     =-0.5 * np.sum(Areafac(self.ndim) * np.pi * binright**(self.ndim - 1) * integralgr * rdelta)

        return S2results

    def Binary(self, ppp, rdelta):
        print ('--------- This is a Binary System ------')

        MAXBIN      = int(self.Boxlength.min() / 2.0 / rdelta)
        S2results   = np.zeros((self.ParticleNumber, self.SnapshotNumber))
        for n in range(self.SnapshotNumber):
            hmatrixinv = np.linalg.inv(self.hmatrix[n])
            for i in range(self.ParticleNumber):
                RIJ          = np.delete(self.Positions[n], i, axis = 0) - self.Positions[n][i]
                matrixij     = np.dot(RIJ, hmatrixinv)
                RIJ          = np.dot(matrixij - np.rint(matrixij) * ppp, self.hmatrix[n]) #remove periodic boundary conditions
                distance     = np.sqrt(np.square(RIJ).sum(axis = 1))
                particletype = np.delete(self.ParticleType[n], i)
                TIJ          = np.c_[particletype, np.zeros_like(particletype) + self.ParticleType[n][i]]
                Countsum     = TIJ.sum(axis = 1)
                
                particlegr          = np.zeros((MAXBIN, 3)) #11 12/21 22
                usedrho             = np.zeros(3)
                Countvalue, BinEdge = np.histogram(distance[Countsum == 2], bins = MAXBIN, range = (0, MAXBIN * rdelta)) #1-1; 
                binleft             = BinEdge[:-1]          #real value of each bin edge, not index
                binright            = BinEdge[1:]           #len(Countvalue) = len(BinEdge) - 1
                Nideal              = Nidealfac(self.ndim) * np.pi * (binright**self.ndim - binleft**self.ndim)
                particlegr[:, 0]    = Countvalue / Nideal / self.rhotype[0]
                usedrho[0]          = self.rhotype[0]
                Countvalue, BinEdge = np.histogram(distance[Countsum == 3], bins = MAXBIN, range = (0, MAXBIN * rdelta)) #1-2; 2-1
                rho12               = self.Type[self.Type != self.ParticleType[n][i]] - 1
                particlegr[:, 1]    = Countvalue / Nideal / self.rhotype[rho12]
                usedrho[1]          = self.rhotype[rho12]
                Countvalue, BinEdge = np.histogram(distance[Countsum == 4], bins = MAXBIN, range = (0, MAXBIN * rdelta)) #2-2
                particlegr[:, 2]    = Countvalue / Nideal / self.rhotype[1]
                usedrho[2]          = self.rhotype[1]
                integralgr          = (particlegr * np.log(particlegr + 1e-12) - (particlegr - 1)) * usedrho[np.newaxis, :]
                integralgr          = integralgr[:, np.any(particlegr, axis = 0)]   #remove zero columns in gr 
                binright           -= 0.5 * rdelta                                  #middle of each bin
                S2results[i, n]     =-0.5 * np.sum(Areafac(self.ndim) * np.pi * binright**(self.ndim - 1) * integralgr.sum(axis = 1) * rdelta)

        return S2results

    def Ternary(self, ppp, rdelta):
        print ('--------- This is a Ternary System ------')

        MAXBIN      = int(self.Boxlength.min() / 2.0 / rdelta)
        S2results   = np.zeros((self.ParticleNumber, self.SnapshotNumber))
        for n in range(self.SnapshotNumber):
            hmatrixinv = np.linalg.inv(self.hmatrix[n])
            for i in range(self.ParticleNumber):
                RIJ          = np.delete(self.Positions[n], i, axis = 0) - self.Positions[n][i]
                matrixij     = np.dot(RIJ, hmatrixinv)
                RIJ          = np.dot(matrixij - np.rint(matrixij) * ppp, self.hmatrix[n]) #remove periodic boundary conditions
                distance     = np.sqrt(np.square(RIJ).sum(axis = 1))
                particletype = np.delete(self.ParticleType[n], i)
                TIJ          = np.c_[particletype, np.zeros_like(particletype) + self.ParticleType[n][i]]
                Countsum     = TIJ.sum(axis = 1)
                Countsub     = np.abs(TIJ[:, 0] - TIJ[:, 1])

                particlegr          = np.zeros((MAXBIN, 6)) #11 12/21 13/31 22 23/32 33
                usedrho             = np.zeros(6)
                Countvalue, BinEdge = np.histogram(distance[Countsum  == 2], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                binleft             = BinEdge[:-1]          #real value of each bin edge, not index
                binright            = BinEdge[1:]           #len(Countvalue) = len(BinEdge) - 1
                Nideal              = Nidealfac(self.ndim) * np.pi * (binright**self.ndim - binleft**self.ndim)
                particlegr[:, 0]    = Countvalue / Nideal / self.rhotype[0] #11
                usedrho[0]          = self.rhotype[0]
                Countvalue, BinEdge = np.histogram(distance[(Countsum == 4) & (Countsub == 0)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                particlegr[:, 1]    = Countvalue / Nideal / self.rhotype[1] #22
                usedrho[1]          = self.rhotype[1]
                Countvalue, BinEdge = np.histogram(distance[Countsum  == 6], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                particlegr[:, 2]    = Countvalue / Nideal / self.rhotype[2] #33
                usedrho[2]          = self.rhotype[2]

                if self.ParticleType[n][i] != 3:
                    Countvalue, BinEdge = np.histogram(distance[Countsum  == 3], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    rho12               = self.Type[self.Type != self.ParticleType[n][i]][0] - 1
                    particlegr[:, 3]    = Countvalue / Nideal / self.rhotype[rho12] #12/21
                    usedrho[3]          = self.rhotype[rho12]

                if self.ParticleType[n][i] != 1:
                    Countvalue, BinEdge = np.histogram(distance[Countsum  == 5], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    rho23               = self.Type[self.Type != self.ParticleType[n][i]][1] - 1
                    particlegr[:, 4]    = Countvalue / Nideal / self.rhotype[rho23] #23/32
                    usedrho[4]          = self.rhotype[rho23]

                if self.ParticleType[n][i] != 2:
                    Countvalue, BinEdge = np.histogram(distance[(Countsum == 4) & (Countsub == 2)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    medium              = self.Type[self.Type != self.ParticleType[n][i]]
                    rho13               = medium[medium != 2] - 1
                    particlegr[:, 5]    = Countvalue / Nideal / self.rhotype[rho13] #13/31
                    usedrho[5]          = self.rhotype[rho13]

                integralgr          = (particlegr * np.log(particlegr + 1e-12) - (particlegr - 1)) * usedrho[np.newaxis, :]
                integralgr          = integralgr[:, np.any(particlegr, axis = 0)]   #remove zero columns in gr 
                binright           -= 0.5 * rdelta                                  #middle of each bin
                S2results[i, n]     =-0.5 * np.sum(Areafac(self.ndim) * np.pi * binright**(self.ndim - 1) * integralgr.sum(axis = 1) * rdelta)

        return S2results

    def Quarternary(self, ppp, rdelta):
        print ('--------- This is a Quarternary System ------')

        MAXBIN      = int(self.Boxlength.min() / 2.0 / rdelta)
        S2results   = np.zeros((self.ParticleNumber, self.SnapshotNumber))
        for n in range(self.SnapshotNumber):
            hmatrixinv = np.linalg.inv(self.hmatrix[n])
            for i in range(self.ParticleNumber):
                RIJ          = np.delete(self.Positions[n], i, axis = 0) - self.Positions[n][i]
                matrixij     = np.dot(RIJ, hmatrixinv)
                RIJ          = np.dot(matrixij - np.rint(matrixij) * ppp, self.hmatrix[n]) #remove periodic boundary conditions
                distance     = np.sqrt(np.square(RIJ).sum(axis = 1))
                particletype = np.delete(self.ParticleType[n], i)
                TIJ          = np.c_[particletype, np.zeros_like(particletype) + self.ParticleType[n][i]]
                Countsum     = TIJ.sum(axis = 1)
                Countsub     = np.abs(TIJ[:, 0] - TIJ[:, 1])

                particlegr          = np.zeros((MAXBIN, 10)) 
                usedrho             = np.zeros(10)
                Countvalue, BinEdge = np.histogram(distance[Countsum == 2], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                binleft             = BinEdge[:-1]          #real value of each bin edge, not index
                binright            = BinEdge[1:]           #len(Countvalue) = len(BinEdge) - 1
                Nideal              = Nidealfac(self.ndim) * np.pi * (binright**self.ndim - binleft**self.ndim)
                particlegr[:, 0]    = Countvalue / Nideal / self.rhotype[0] #11
                usedrho[0]          = self.rhotype[0]
                Countvalue, BinEdge = np.histogram(distance[(Countsum  == 4) & (Countsub == 0)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                particlegr[:, 1]    = Countvalue / Nideal / self.rhotype[1] #22
                usedrho[1]          = self.rhotype[1]
                Countvalue, BinEdge = np.histogram(distance[(Countsum  == 6) & (Countsub == 0)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                particlegr[:, 2]    = Countvalue / Nideal / self.rhotype[2] #33
                usedrho[2]          = self.rhotype[2]
                Countvalue, BinEdge = np.histogram(distance[Countsum  == 8], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                particlegr[:, 3]    = Countvalue / Nideal / self.rhotype[3] #44
                usedrho[3]          = self.rhotype[3]

                if self.ParticleType[n][i] in [1, 2]:
                    Countvalue, BinEdge = np.histogram(distance[Countsum  == 3], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    rho12               = self.Type[self.Type != self.ParticleType[n][i]][0] - 1
                    particlegr[:, 4]    = Countvalue / Nideal / self.rhotype[rho12] #12
                    usedrho[4]          = self.rhotype[rho12]

                if self.ParticleType[n][i] in [1, 3]:
                    Countvalue, BinEdge = np.histogram(distance[(Countsum  == 4) & (Countsub == 2)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    medium              = self.Type[self.Type != self.ParticleType[n][i]][:2]
                    rho13               = medium[medium != 2] - 1
                    particlegr[:, 5]    = Countvalue / Nideal / self.rhotype[rho13] #13
                    usedrho[5]          = self.rhotype[rho13]

                if self.ParticleType[n][i] in [1, 4]:
                    Countvalue, BinEdge = np.histogram(distance[Countsub  == 3], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    medium              = self.Type[self.Type != self.ParticleType[n][i]]
                    rho14               = medium[(medium != 2) & (medium != 3)] - 1
                    particlegr[:, 6]    = Countvalue / Nideal / self.rhotype[rho14] #14
                    usedrho[6]          = self.rhotype[rho14]

                if self.ParticleType[n][i] in [2, 3]:
                    Countvalue, BinEdge = np.histogram(distance[(Countsum  == 5) & (Countsub == 1)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    rho23               = self.Type[self.Type != self.ParticleType[n][i]][1] - 1
                    particlegr[:, 7]    = Countvalue / Nideal / self.rhotype[rho23] #23
                    usedrho[7]          = self.rhotype[rho23]

                if self.ParticleType[n][i] in [2, 4]:
                    Countvalue, BinEdge = np.histogram(distance[(Countsum  == 6) & (Countsub == 2)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    medium              = self.Type[self.Type != self.ParticleType[n][i]][1:]
                    rho24               = medium[medium != 3] - 1
                    particlegr[:, 8]    = Countvalue / Nideal / self.rhotype[rho24] #24
                    usedrho[8]          = self.rhotype[rho24]

                if self.ParticleType[n][i] in [3, 4]:
                    Countvalue, BinEdge = np.histogram(distance[Countsum  == 7], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    rho34               = self.Type[self.Type != self.ParticleType[n][i]][-1] - 1
                    particlegr[:, 9]    = Countvalue / Nideal / self.rhotype[rho34] #34
                    usedrho[9]          = self.rhotype[rho34]

                integralgr          = (particlegr * np.log(particlegr + 1e-12) - (particlegr - 1)) * usedrho[np.newaxis, :]
                integralgr          = integralgr[:, np.any(particlegr, axis = 0)]   #remove zero columns in gr 
                binright           -= 0.5 * rdelta                                  #middle of each bin
                S2results[i, n]     =-0.5 * np.sum(Areafac(self.ndim) * np.pi * binright**(self.ndim - 1) * integralgr.sum(axis = 1) * rdelta)

        return S2results

    def Quinary(self, ppp, rdelta):
        print ('--------- This is a Quinary System ------')

        MAXBIN      = int(self.Boxlength.min() / 2.0 / rdelta)
        S2results   = np.zeros((self.ParticleNumber, self.SnapshotNumber))
        for n in range(self.SnapshotNumber):
            hmatrixinv = np.linalg.inv(self.hmatrix[n])
            for i in range(self.ParticleNumber):
                RIJ          = np.delete(self.Positions[n], i, axis = 0) - self.Positions[n][i]
                matrixij     = np.dot(RIJ, hmatrixinv)
                RIJ          = np.dot(matrixij - np.rint(matrixij) * ppp, self.hmatrix[n]) #remove periodic boundary conditions
                distance     = np.sqrt(np.square(RIJ).sum(axis = 1))
                particletype = np.delete(self.ParticleType[n], i)
                TIJ          = np.c_[particletype, np.zeros_like(particletype) + self.ParticleType[n][i]]
                Countsum     = TIJ.sum(axis = 1)
                Countsub     = np.abs(TIJ[:, 0] - TIJ[:, 1])

                particlegr          = np.zeros((MAXBIN, 15)) 
                usedrho             = np.zeros(15)
                Countvalue, BinEdge = np.histogram(distance[Countsum == 2], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                binleft             = BinEdge[:-1]          #real value of each bin edge, not index
                binright            = BinEdge[1:]           #len(Countvalue) = len(BinEdge) - 1
                Nideal              = Nidealfac(self.ndim) * np.pi * (binright**self.ndim - binleft**self.ndim)
                particlegr[:, 0]    = Countvalue / Nideal / self.rhotype[0] #11
                usedrho[0]          = self.rhotype[0]
                Countvalue, BinEdge = np.histogram(distance[(Countsum  == 4) & (Countsub == 0)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                particlegr[:, 1]    = Countvalue / Nideal / self.rhotype[1] #22
                usedrho[1]          = self.rhotype[1]
                Countvalue, BinEdge = np.histogram(distance[(Countsum  == 6) & (Countsub == 0)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                particlegr[:, 2]    = Countvalue / Nideal / self.rhotype[2] #33
                usedrho[2]          = self.rhotype[2]
                Countvalue, BinEdge = np.histogram(distance[(Countsum  == 8) & (Countsub == 0)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                particlegr[:, 3]    = Countvalue / Nideal / self.rhotype[3] #44
                usedrho[3]          = self.rhotype[3]
                Countvalue, BinEdge = np.histogram(distance[Countsum  == 10], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                particlegr[:, 4]    = Countvalue / Nideal / self.rhotype[4] #55
                usedrho[4]          = self.rhotype[4]

                if self.ParticleType[n][i] in [1, 2]:
                    Countvalue, BinEdge = np.histogram(distance[Countsum  == 3], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    rho12               = self.Type[self.Type != self.ParticleType[n][i]][0] - 1
                    particlegr[:, 5]    = Countvalue / Nideal / self.rhotype[rho12] #12
                    usedrho[5]          = self.rhotype[rho12]

                if self.ParticleType[n][i] in [1, 3]:
                    Countvalue, BinEdge = np.histogram(distance[(Countsum  == 4) & (Countsub == 2)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    medium              = self.Type[self.Type != self.ParticleType[n][i]][:2]
                    rho13               = medium[medium != 2] - 1
                    particlegr[:, 6]    = Countvalue / Nideal / self.rhotype[rho13] #13
                    usedrho[6]          = self.rhotype[rho13]

                if self.ParticleType[n][i] in [1, 4]:
                    Countvalue, BinEdge = np.histogram(distance[(Countsum  == 5) & (Countsub == 3)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    medium              = self.Type[self.Type != self.ParticleType[n][i]][:3]
                    rho14               = medium[(medium != 2) & (medium != 3)] - 1
                    particlegr[:, 7]    = Countvalue / Nideal / self.rhotype[rho14] #14
                    usedrho[7]          = self.rhotype[rho14]

                if self.ParticleType[n][i] in [1, 5]:
                    Countvalue, BinEdge = np.histogram(distance[(Countsum  == 6) & (Countsub == 4)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    medium              = self.Type[self.Type != self.ParticleType[n][i]]
                    rho15               = medium[(medium != 2) & (medium != 3) & (medium != 4)] - 1
                    particlegr[:, 8]    = Countvalue / Nideal / self.rhotype[rho15] #15
                    usedrho[8]          = self.rhotype[rho15]

                if self.ParticleType[n][i] in [2, 3]:
                    Countvalue, BinEdge = np.histogram(distance[(Countsum  == 5) & (Countsub == 1)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    rho23               = self.Type[self.Type != self.ParticleType[n][i]][1] - 1
                    particlegr[:,9]     = Countvalue / Nideal / self.rhotype[rho23] #23
                    usedrho[9]          = self.rhotype[rho23]

                if self.ParticleType[n][i] in [2, 4]:
                    Countvalue, BinEdge = np.histogram(distance[(Countsum  == 6) & (Countsub == 2)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    medium              = self.Type[self.Type != self.ParticleType[n][i]][1:3]
                    rho24               = medium[medium != 3] - 1
                    particlegr[:,10]    = Countvalue / Nideal / self.rhotype[rho24] #24
                    usedrho[10]         = self.rhotype[rho24]

                if self.ParticleType[n][i] in [2, 5]:
                    Countvalue, BinEdge = np.histogram(distance[(Countsum  == 7) & (Countsub == 3)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    medium              = self.Type[self.Type != self.ParticleType[n][i]][1:]
                    rho25               = medium[(medium != 3) & (medium != 4)] - 1
                    particlegr[:,11]    = Countvalue / Nideal / self.rhotype[rho25] #25
                    usedrho[11]         = self.rhotype[rho25]

                if self.ParticleType[n][i] in [3, 4]:
                    Countvalue, BinEdge = np.histogram(distance[(Countsum  == 7) & (Countsub == 1)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    rho34               = self.Type[self.Type != self.ParticleType[n][i]][2] - 1
                    particlegr[:,12]    = Countvalue / Nideal / self.rhotype[rho34] #34
                    usedrho[12]         = self.rhotype[rho34]

                if self.ParticleType[n][i] in [3, 5]:
                    Countvalue, BinEdge = np.histogram(distance[(Countsum  == 8) & (Countsub == 2)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    medium              = self.Type[self.Type != self.ParticleType[n][i]][2:]
                    rho35               = medium[medium != 4] - 1
                    particlegr[:,13]    = Countvalue / Nideal / self.rhotype[rho35] #35
                    usedrho[13]         = self.rhotype[rho35]

                if self.ParticleType[n][i] in [4, 5]:
                    Countvalue, BinEdge = np.histogram(distance[Countsum  == 9], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    rho45               = self.Type[self.Type != self.ParticleType[n][i]][-1] - 1
                    particlegr[:,14]    = Countvalue / Nideal / self.rhotype[rho45] #45
                    usedrho[14]         = self.rhotype[rho45]

                integralgr          = (particlegr * np.log(particlegr + 1e-12) - (particlegr - 1)) * usedrho[np.newaxis, :]
                integralgr          = integralgr[:, np.any(particlegr, axis = 0)]   #remove zero columns in gr 
                binright           -= 0.5 * rdelta                                  #middle of each bin
                S2results[i, n]     =-0.5 * np.sum(Areafac(self.ndim) * np.pi * binright**(self.ndim - 1) * integralgr.sum(axis = 1) * rdelta)

        return S2results

class S2AVE:
    """ Compute pair entropy S2 by Averaging particle gr over different snapshots """

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
        d = readdump(self.inputfile, self.ndim, self.filetype, self.moltypes)
        d.read_onefile()

        self.TimeStep     = d.TimeStep[1] - d.TimeStep[0]
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
        self.hmatrix    = d.hmatrix
        self.Volume     = np.prod(self.Boxlength)
        self.rhototal   = self.ParticleNumber / self.Volume #number density of the total system
        self.typecounts = np.unique(self.ParticleType[0], return_counts = True) 
        self.Type       = self.typecounts[0]
        self.TypeNumber = self.typecounts[1]
        self.rhotype    = self.TypeNumber / self.Volume
        print ('Particle Type:', self.Type)
        print ('Particle TypeNumber:', self.TypeNumber)
        if np.sum(self.TypeNumber) != self.ParticleNumber:
            print ('Warning: ****** Sum of Indivdual Types is Not the Total Amount*******')

    def getS2(self, ppp, avetime, rdelta = 0.01, dt = 0.002, outputfile = ''):
        """ Get Particle-level S2 by averaging particle gr over different snapshots
            
            outputfile is file name storing outputs
            ppp is periodic boundary conditions, set 1 for yes and 0 for no, should be a list
            avetime is the time scale used to average particle gr, in time unit
            rdelta is the bin size of gr, default is 0.01
            dt is timestep of a simulation, default is 0.002
        """

        if len(self.Type) == 1:
            S2results      = self.Unary(ppp, rdelta, avetime, dt)
        if len(self.Type) == 2: 
            S2results      = self.Binary(ppp, rdelta, avetime, dt)
        if len(self.Type) == 3: 
            S2results      = self.Ternary(ppp, rdelta, avetime, dt)
        if len(self.Type) == 4: 
            S2results      = self.Quarternary(ppp, rdelta, avetime, dt)
        if len(self.Type) == 5: 
            S2results      = self.Quinary(ppp, rdelta, avetime, dt)

        results    = np.column_stack((np.arange(self.ParticleNumber) + 1, S2results))
        names      = 'id   S2_of_each_snapshot'
        fileformat = '%d ' + '%.6f ' * S2results.shape[1]
        if outputfile:
            np.savetxt(outputfile, results, fmt = fileformat, header = names, comments = '')

        print ('---------- Get S2 results over ---------')
        return S2results, names

    def spatialcorr(self, outputs2, ppp, avetime, rdelta = 0.01, dt = 0.002, outputcorr = ''):
        """ Calculate Spatial Correlation of S2 that is obtained by averaging particle gr
            
            Excuting this function will excute the function getS2() first, so S2 will be output
            outputcorr is file name storing outputs of spatial correlation of S2
            outputs2 is file name storing outputs of S2
            ppp is periodic boundary conditions, set 1 for yes and 0 for no, should be a list
            avetime is the time scale used to average gr, in time unit
            rdelta is the bin size of gr, default is 0.01
            dt is timestep of a simulation, default is 0.002
        """
        print ('--------- Calculating Spatial Correlation of S2 ------')

        timeaves2      = self.getS2(outputs2, ppp, avetime, rdelta, dt, results_path).T 
        MAXBIN         = int(self.Boxlength.min() / 2.0 / rdelta)
        grresults      = np.zeros((MAXBIN, 3))
        SnapshotNumber = timeaves2.shape[0]
        Positions      = self.Positions[:SnapshotNumber]
        for n in range(SnapshotNumber):
            hmatrixinv = np.linalg.inv(self.hmatrix[n])
            for i in range(self.ParticleNumber - 1):
                RIJ      = Positions[n][i+1:] - Positions[n][i]
                #periodic = np.where(np.abs(RIJ / self.Boxlength[np.newaxis, :]) > 0.50, np.sign(RIJ), 0).astype(np.int)
                #RIJ -= self.Boxlength * periodic * ppp    #remove PBC
                matrixij = np.dot(RIJ, hmatrixinv)
                RIJ      = np.dot(matrixij - np.rint(matrixij) * ppp, self.hmatrix[n]) #remove PBC
                distance = np.sqrt(np.square(RIJ).sum(axis = 1))
                Countvalue, BinEdge = np.histogram(distance, bins = MAXBIN, range = (0, MAXBIN * rdelta))
                grresults[:, 0]    += Countvalue
                S2IJ                = timeaves2[n, i + 1:] * timeaves2[n, i]
                Countvalue, BinEdge = np.histogram(distance, bins = MAXBIN, range = (0, MAXBIN * rdelta), weights = S2IJ)
                grresults[:, 1]    += Countvalue

        binleft          = BinEdge[:-1]   #real value of each bin edge, not index
        binright         = BinEdge[1:]    #len(Countvalue) = len(BinEdge) - 1
        Nideal           = Nidealfac(self.ndim) * np.pi * (binright**self.ndim - binleft**self.ndim) 
        grresults       *= 2 / self.ParticleNumber / SnapshotNumber / (Nideal[:, np.newaxis] * self.rhototal)
        grresults[:, 2]  = np.where(grresults[:, 0] != 0, grresults[:, 1] / grresults[:, 0], np.nan)
        binright        -= 0.5 * rdelta #middle of each bin
        results          = np.column_stack((binright, grresults))
        names            = 'r   g(r)   gs2(r)   gs2/gr'
        if outputcorr:
            np.savetxt(outputcorr, results, fmt='%.6f', header = names, comments = '')

        print ('---------- Get gs2(r) results over ---------')
        return results, names

    def Unary(self, ppp, rdelta, avetime, dt):
        print ('--------- This is a Unary System ------')

        avetime     = int(avetime / dt / self.TimeStep)
        MAXBIN      = int(self.Boxlength.min() / 2.0 / rdelta)
        S2results   = np.zeros((self.ParticleNumber, self.SnapshotNumber - avetime))
        particlegr  = np.zeros((self.ParticleNumber, self.SnapshotNumber, MAXBIN)) #11
        for n in range(self.SnapshotNumber):
            hmatrixinv = np.linalg.inv(self.hmatrix[n])
            for i in range(self.ParticleNumber):
                RIJ          = np.delete(self.Positions[n], i, axis = 0) - self.Positions[n][i]
                matrixij     = np.dot(RIJ, hmatrixinv)
                RIJ          = np.dot(matrixij - np.rint(matrixij) * ppp, self.hmatrix[n]) #remove periodic boundary conditions
                distance     = np.sqrt(np.square(RIJ).sum(axis = 1))
                Countvalue, BinEdge = np.histogram(distance, bins = MAXBIN, range = (0, MAXBIN * rdelta)) #1-1; 
                if n == 0 and i == 0:
                    binleft         = BinEdge[:-1]          #real value of each bin edge, not index
                    binright        = BinEdge[1:]           #len(Countvalue) = len(BinEdge) - 1
                    Nideal          = Nidealfac(self.ndim) * np.pi * (binright**self.ndim - binleft**self.ndim)
                particlegr[i, n]    = Countvalue / Nideal / self.rhototal
        
        binright -= 0.5 * rdelta #middle of each bin
        for n in range(self.SnapshotNumber - avetime):
            for i in range(self.ParticleNumber):
                aveparticlegr       = particlegr[i, n: n + avetime + 1].mean(axis = 0)
                integralgr          = (aveparticlegr * np.log(aveparticlegr + 1e-12) - (aveparticlegr - 1)) * self.rhototal
                S2results[i, n]     =-0.5 * np.sum(Areafac(self.ndim) * np.pi * binright**(self.ndim - 1) * integralgr * rdelta)

        return S2results

    def Binary(self, ppp, rdelta, avetime, dt):
        print ('--------- This is a Binary System ------')

        avetime     = int(avetime / dt / self.TimeStep)
        MAXBIN      = int(self.Boxlength.min() / 2.0 / rdelta)
        S2results   = np.zeros((self.ParticleNumber, self.SnapshotNumber - avetime))
        particlegr  = np.zeros((self.ParticleNumber, self.SnapshotNumber, MAXBIN, 3))
        usedrho     = np.zeros((self.ParticleNumber, self.SnapshotNumber, 3))
        for n in range(self.SnapshotNumber):
            hmatrixinv = np.linalg.inv(self.hmatrix[n])
            for i in range(self.ParticleNumber):
                RIJ          = np.delete(self.Positions[n], i, axis = 0) - self.Positions[n][i]
                matrixij     = np.dot(RIJ, hmatrixinv)
                RIJ          = np.dot(matrixij - np.rint(matrixij) * ppp, self.hmatrix[n]) #remove periodic boundary conditions
                distance     = np.sqrt(np.square(RIJ).sum(axis = 1))
                particletype = np.delete(self.ParticleType[n], i)
                TIJ          = np.c_[particletype, np.zeros_like(particletype) + self.ParticleType[n][i]]
                Countsum     = TIJ.sum(axis = 1)
                
                Countvalue, BinEdge = np.histogram(distance[Countsum == 2], bins = MAXBIN, range = (0, MAXBIN * rdelta)) #1-1; 
                if n == 0 and i == 0:
                    binleft             = BinEdge[:-1]          #real value of each bin edge, not index
                    binright            = BinEdge[1:]           #len(Countvalue) = len(BinEdge) - 1
                    Nideal              = Nidealfac(self.ndim) * np.pi * (binright**self.ndim - binleft**self.ndim)
                particlegr[i, n, :, 0]  = Countvalue / Nideal / self.rhotype[0]
                usedrho[i, n, 0]        = self.rhotype[0]
                Countvalue, BinEdge     = np.histogram(distance[Countsum == 3], bins = MAXBIN, range = (0, MAXBIN * rdelta)) #1-2; 2-1
                rho12                   = self.Type[self.Type != self.ParticleType[n][i]] - 1
                particlegr[i, n, :, 1]  = Countvalue / Nideal / self.rhotype[rho12]
                usedrho[i, n, 1]        = self.rhotype[rho12]
                Countvalue, BinEdge     = np.histogram(distance[Countsum == 4], bins = MAXBIN, range = (0, MAXBIN * rdelta)) #2-2
                particlegr[i, n, :, 2]  = Countvalue / Nideal / self.rhotype[1]
                usedrho[i, n, 2]        = self.rhotype[1]

        binright -= 0.5 * rdelta #middle of each bin
        for n in range(self.SnapshotNumber - avetime):
            for i in range(self.ParticleNumber):
                aveparticlegr       = particlegr[i, n: n + avetime + 1].mean(axis = 0)
                integralgr          = (aveparticlegr * np.log(aveparticlegr + 1e-12) - (aveparticlegr - 1)) * usedrho[i, n][np.newaxis, :]
                integralgr          = integralgr[:, np.any(aveparticlegr, axis = 0)]   #remove zero columns in gr 
                S2results[i, n]     =-0.5 * np.sum(Areafac(self.ndim) * np.pi * binright**(self.ndim - 1) * integralgr.sum(axis = 1) * rdelta)

        return S2results

    def Ternary(self, ppp, rdelta, avetime, dt):
        print ('--------- This is a Ternary System ------')

        avetime     = int(avetime / dt / self.TimeStep)
        MAXBIN      = int(self.Boxlength.min() / 2.0 / rdelta)
        S2results   = np.zeros((self.ParticleNumber, self.SnapshotNumber - avetime))
        particlegr  = np.zeros((self.ParticleNumber, self.SnapshotNumber, MAXBIN, 6))
        usedrho     = np.zeros((self.ParticleNumber, self.SnapshotNumber, 6))
        for n in range(self.SnapshotNumber):
            hmatrixinv = np.linalg.inv(self.hmatrix[n])
            for i in range(self.ParticleNumber):
                RIJ          = np.delete(self.Positions[n], i, axis = 0) - self.Positions[n][i]
                matrixij     = np.dot(RIJ, hmatrixinv)
                RIJ          = np.dot(matrixij - np.rint(matrixij) * ppp, self.hmatrix[n]) #remove periodic boundary conditions
                distance     = np.sqrt(np.square(RIJ).sum(axis = 1))
                particletype = np.delete(self.ParticleType[n], i)
                TIJ          = np.c_[particletype, np.zeros_like(particletype) + self.ParticleType[n][i]]
                Countsum     = TIJ.sum(axis = 1)
                Countsub     = np.abs(TIJ[:, 0] - TIJ[:, 1])

                Countvalue, BinEdge = np.histogram(distance[Countsum  == 2], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                if n == 0 and i == 0:
                    binleft            = BinEdge[:-1]          #real value of each bin edge, not index
                    binright           = BinEdge[1:]           #len(Countvalue) = len(BinEdge) - 1
                    Nideal             = Nidealfac(self.ndim) * np.pi * (binright**self.ndim - binleft**self.ndim)
                particlegr[i, n, :, 0] = Countvalue / Nideal / self.rhotype[0] #11
                usedrho[i, n, 0]       = self.rhotype[0]
                Countvalue, BinEdge    = np.histogram(distance[(Countsum == 4) & (Countsub == 0)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                particlegr[i, n, :, 1] = Countvalue / Nideal / self.rhotype[1] #22
                usedrho[i, n, 1]       = self.rhotype[1]
                Countvalue, BinEdge    = np.histogram(distance[Countsum  == 6], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                particlegr[i, n, :, 2] = Countvalue / Nideal / self.rhotype[2] #33
                usedrho[i, n, 2]       = self.rhotype[2]

                if self.ParticleType[n][i] != 3:
                    Countvalue, BinEdge    = np.histogram(distance[Countsum  == 3], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    rho12                  = self.Type[self.Type != self.ParticleType[n][i]][0] - 1
                    particlegr[i, n, :, 3] = Countvalue / Nideal / self.rhotype[rho12] #12/21
                    usedrho[i, n, 3]       = self.rhotype[rho12]

                if self.ParticleType[n][i] != 1:
                    Countvalue, BinEdge    = np.histogram(distance[Countsum  == 5], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    rho23                  = self.Type[self.Type != self.ParticleType[n][i]][1] - 1
                    particlegr[i, n, :, 4] = Countvalue / Nideal / self.rhotype[rho23] #23/32
                    usedrho[i, n, 4]       = self.rhotype[rho23]

                if self.ParticleType[n][i] != 2:
                    Countvalue, BinEdge    = np.histogram(distance[(Countsum == 4) & (Countsub == 2)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    medium                 = self.Type[self.Type != self.ParticleType[n][i]]
                    rho13                  = medium[medium != 2] - 1
                    particlegr[i, n, :, 5] = Countvalue / Nideal / self.rhotype[rho13] #13/31
                    usedrho[i, n, 5]       = self.rhotype[rho13]

        binright -= 0.5 * rdelta #middle of each bin
        for n in range(self.SnapshotNumber - avetime):
            for i in range(self.ParticleNumber):
                aveparticlegr       = particlegr[i, n: n + avetime + 1].mean(axis = 0)
                integralgr          = (aveparticlegr * np.log(aveparticlegr + 1e-12) - (aveparticlegr - 1)) * usedrho[i, n][np.newaxis, :]
                integralgr          = integralgr[:, np.any(aveparticlegr, axis = 0)]   #remove zero columns in gr 
                S2results[i, n]     =-0.5 * np.sum(Areafac(self.ndim) * np.pi * binright**(self.ndim - 1) * integralgr.sum(axis = 1) * rdelta)

        return S2results

    def Quarternary(self, ppp, rdelta):
        print ('--------- This is a Quarternary System ------')

        avetime     = int(avetime / dt / self.TimeStep)
        MAXBIN      = int(self.Boxlength.min() / 2.0 / rdelta)
        S2results   = np.zeros((self.ParticleNumber, self.SnapshotNumber - avetime))
        particlegr  = np.zeros((self.ParticleNumber, self.SnapshotNumber, MAXBIN, 10))
        usedrho     = np.zeros((self.ParticleNumber, self.SnapshotNumber, 10))
        for n in range(self.SnapshotNumber):
            hmatrixinv = np.linalg.inv(self.hmatrix[n])
            for i in range(self.ParticleNumber):
                RIJ          = np.delete(self.Positions[n], i, axis = 0) - self.Positions[n][i]
                matrixij     = np.dot(RIJ, hmatrixinv)
                RIJ          = np.dot(matrixij - np.rint(matrixij) * ppp, self.hmatrix[n]) #remove periodic boundary conditions
                distance     = np.sqrt(np.square(RIJ).sum(axis = 1))
                particletype = np.delete(self.ParticleType[n], i)
                TIJ          = np.c_[particletype, np.zeros_like(particletype) + self.ParticleType[n][i]]
                Countsum     = TIJ.sum(axis = 1)
                Countsub     = np.abs(TIJ[:, 0] - TIJ[:, 1])

                Countvalue, BinEdge = np.histogram(distance[Countsum == 2], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                if n == 0 and i == 0:
                    binleft            = BinEdge[:-1]          #real value of each bin edge, not index
                    binright           = BinEdge[1:]           #len(Countvalue) = len(BinEdge) - 1
                    Nideal             = Nidealfac(self.ndim) * np.pi * (binright**self.ndim - binleft**self.ndim)
                particlegr[i, n, :, 0] = Countvalue / Nideal / self.rhotype[0] #11
                usedrho[i, n, 0]       = self.rhotype[0]
                Countvalue, BinEdge    = np.histogram(distance[(Countsum  == 4) & (Countsub == 0)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                particlegr[i, n, :, 1] = Countvalue / Nideal / self.rhotype[1] #22
                usedrho[i, n, 1]       = self.rhotype[1]
                Countvalue, BinEdge    = np.histogram(distance[(Countsum  == 6) & (Countsub == 0)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                particlegr[i, n, :, 2] = Countvalue / Nideal / self.rhotype[2] #33
                usedrho[i, n, 2]       = self.rhotype[2]
                Countvalue, BinEdge    = np.histogram(distance[Countsum  == 8], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                particlegr[i, n, :, 3] = Countvalue / Nideal / self.rhotype[3] #44
                usedrho[i, n, 3]       = self.rhotype[3]

                if self.ParticleType[n][i] in [1, 2]:
                    Countvalue, BinEdge    = np.histogram(distance[Countsum  == 3], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    rho12                  = self.Type[self.Type != self.ParticleType[n][i]][0] - 1
                    particlegr[i, n, :, 4] = Countvalue / Nideal / self.rhotype[rho12] #12
                    usedrho[i, n, 4]       = self.rhotype[rho12]

                if self.ParticleType[n][i] in [1, 3]:
                    Countvalue, BinEdge    = np.histogram(distance[(Countsum  == 4) & (Countsub == 2)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    medium                 = self.Type[self.Type != self.ParticleType[n][i]][:2]
                    rho13                  = medium[medium != 2] - 1
                    particlegr[i, n, :, 5] = Countvalue / Nideal / self.rhotype[rho13] #13
                    usedrho[i, n, 5]       = self.rhotype[rho13]

                if self.ParticleType[n][i] in [1, 4]:
                    Countvalue, BinEdge    = np.histogram(distance[Countsub  == 3], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    medium                 = self.Type[self.Type != self.ParticleType[n][i]]
                    rho14                  = medium[(medium != 2) & (medium != 3)] - 1
                    particlegr[i, n, :, 6] = Countvalue / Nideal / self.rhotype[rho14] #14
                    usedrho[i, n, 6]       = self.rhotype[rho14]

                if self.ParticleType[n][i] in [2, 3]:
                    Countvalue, BinEdge    = np.histogram(distance[(Countsum  == 5) & (Countsub == 1)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    rho23                  = self.Type[self.Type != self.ParticleType[n][i]][1] - 1
                    particlegr[i, n, :, 7] = Countvalue / Nideal / self.rhotype[rho23] #23
                    usedrho[i, n, 7]       = self.rhotype[rho23]

                if self.ParticleType[n][i] in [2, 4]:
                    Countvalue, BinEdge    = np.histogram(distance[(Countsum  == 6) & (Countsub == 2)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    medium                 = self.Type[self.Type != self.ParticleType[n][i]][1:]
                    rho24                  = medium[medium != 3] - 1
                    particlegr[i, n, :, 8] = Countvalue / Nideal / self.rhotype[rho24] #24
                    usedrho[i, n, 8]       = self.rhotype[rho24]

                if self.ParticleType[n][i] in [3, 4]:
                    Countvalue, BinEdge    = np.histogram(distance[Countsum  == 7], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    rho34                  = self.Type[self.Type != self.ParticleType[n][i]][-1] - 1
                    particlegr[i, n, :, 9] = Countvalue / Nideal / self.rhotype[rho34] #34
                    usedrho[i, n, 9]       = self.rhotype[rho34]

        binright -= 0.5 * rdelta #middle of each bin
        for n in range(self.SnapshotNumber - avetime):
            for i in range(self.ParticleNumber):
                aveparticlegr       = particlegr[i, n: n + avetime + 1].mean(axis = 0)
                integralgr          = (aveparticlegr * np.log(aveparticlegr + 1e-12) - (aveparticlegr - 1)) * usedrho[i, n][np.newaxis, :]
                integralgr          = integralgr[:, np.any(aveparticlegr, axis = 0)]   #remove zero columns in gr 
                S2results[i, n]     =-0.5 * np.sum(Areafac(self.ndim) * np.pi * binright**(self.ndim - 1) * integralgr.sum(axis = 1) * rdelta)

        return S2results

    def Quinary(self, ppp, rdelta):
        print ('--------- This is a Quinary System ------')

        avetime     = int(avetime / dt / self.TimeStep)
        MAXBIN      = int(self.Boxlength.min() / 2.0 / rdelta)
        S2results   = np.zeros((self.ParticleNumber, self.SnapshotNumber - avetime))
        particlegr  = np.zeros((self.ParticleNumber, self.SnapshotNumber, MAXBIN, 15))
        usedrho     = np.zeros((self.ParticleNumber, self.SnapshotNumber, 15))
        for n in range(self.SnapshotNumber):
            hmatrixinv = np.linalg.inv(self.hmatrix[n])
            for i in range(self.ParticleNumber):
                RIJ          = np.delete(self.Positions[n], i, axis = 0) - self.Positions[n][i]
                matrixij     = np.dot(RIJ, hmatrixinv)
                RIJ          = np.dot(matrixij - np.rint(matrixij) * ppp, self.hmatrix[n]) #remove periodic boundary conditions
                distance     = np.sqrt(np.square(RIJ).sum(axis = 1))
                particletype = np.delete(self.ParticleType[n], i)
                TIJ          = np.c_[particletype, np.zeros_like(particletype) + self.ParticleType[n][i]]
                Countsum     = TIJ.sum(axis = 1)
                Countsub     = np.abs(TIJ[:, 0] - TIJ[:, 1])

                Countvalue, BinEdge = np.histogram(distance[Countsum == 2], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                if n == 0 and i == 0:
                    binleft            = BinEdge[:-1]          #real value of each bin edge, not index
                    binright           = BinEdge[1:]           #len(Countvalue) = len(BinEdge) - 1
                    Nideal             = Nidealfac(self.ndim) * np.pi * (binright**self.ndim - binleft**self.ndim)
                particlegr[i, n, :, 0] = Countvalue / Nideal / self.rhotype[0] #11
                usedrho[i, n, 0]       = self.rhotype[0]
                Countvalue, BinEdge    = np.histogram(distance[(Countsum  == 4) & (Countsub == 0)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                particlegr[i, n, :, 1] = Countvalue / Nideal / self.rhotype[1] #22
                usedrho[i, n, 1]       = self.rhotype[1]
                Countvalue, BinEdge    = np.histogram(distance[(Countsum  == 6) & (Countsub == 0)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                particlegr[i, n, :, 2] = Countvalue / Nideal / self.rhotype[2] #33
                usedrho[i, n, 2]       = self.rhotype[2]
                Countvalue, BinEdge    = np.histogram(distance[(Countsum  == 8) & (Countsub == 0)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                particlegr[i, n, :, 3] = Countvalue / Nideal / self.rhotype[3] #44
                usedrho[i, n, 3]       = self.rhotype[3]
                Countvalue, BinEdge    = np.histogram(distance[Countsum  == 10], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                particlegr[i, n, :, 4] = Countvalue / Nideal / self.rhotype[4] #55
                usedrho[i, n, 4]       = self.rhotype[4]

                if self.ParticleType[n][i] in [1, 2]:
                    Countvalue, BinEdge    = np.histogram(distance[Countsum  == 3], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    rho12                  = self.Type[self.Type != self.ParticleType[n][i]][0] - 1
                    particlegr[i, n, :, 5] = Countvalue / Nideal / self.rhotype[rho12] #12
                    usedrho[i, n, 5]       = self.rhotype[rho12]

                if self.ParticleType[n][i] in [1, 3]:
                    Countvalue, BinEdge    = np.histogram(distance[(Countsum  == 4) & (Countsub == 2)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    medium                 = self.Type[self.Type != self.ParticleType[n][i]][:2]
                    rho13                  = medium[medium != 2] - 1
                    particlegr[i, n, :, 6] = Countvalue / Nideal / self.rhotype[rho13] #13
                    usedrho[i, n, 6]       = self.rhotype[rho13]

                if self.ParticleType[n][i] in [1, 4]:
                    Countvalue, BinEdge    = np.histogram(distance[(Countsum  == 5) & (Countsub == 3)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    medium                 = self.Type[self.Type != self.ParticleType[n][i]][:3]
                    rho14                  = medium[(medium != 2) & (medium != 3)] - 1
                    particlegr[i, n, :, 7] = Countvalue / Nideal / self.rhotype[rho14] #14
                    usedrho[i, n, 7]       = self.rhotype[rho14]

                if self.ParticleType[n][i] in [1, 5]:
                    Countvalue, BinEdge    = np.histogram(distance[(Countsum  == 6) & (Countsub == 4)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    medium                 = self.Type[self.Type != self.ParticleType[n][i]]
                    rho15                  = medium[(medium != 2) & (medium != 3) & (medium != 4)] - 1
                    particlegr[i, n, :, 8] = Countvalue / Nideal / self.rhotype[rho15] #15
                    usedrho[i, n, 8]       = self.rhotype[rho15]

                if self.ParticleType[n][i] in [2, 3]:
                    Countvalue, BinEdge    = np.histogram(distance[(Countsum  == 5) & (Countsub == 1)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    rho23                  = self.Type[self.Type != self.ParticleType[n][i]][1] - 1
                    particlegr[i, n, :,9]  = Countvalue / Nideal / self.rhotype[rho23] #23
                    usedrho[i, n, 9]       = self.rhotype[rho23]

                if self.ParticleType[n][i] in [2, 4]:
                    Countvalue, BinEdge    = np.histogram(distance[(Countsum  == 6) & (Countsub == 2)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    medium                 = self.Type[self.Type != self.ParticleType[n][i]][1:3]
                    rho24                  = medium[medium != 3] - 1
                    particlegr[i, n, :,10] = Countvalue / Nideal / self.rhotype[rho24] #24
                    usedrho[i, n, 10]      = self.rhotype[rho24]

                if self.ParticleType[n][i] in [2, 5]:
                    Countvalue, BinEdge    = np.histogram(distance[(Countsum  == 7) & (Countsub == 3)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    medium                 = self.Type[self.Type != self.ParticleType[n][i]][1:]
                    rho25                  = medium[(medium != 3) & (medium != 4)] - 1
                    particlegr[i, n, :,11] = Countvalue / Nideal / self.rhotype[rho25] #25
                    usedrho[i, n, 11]      = self.rhotype[rho25]

                if self.ParticleType[n][i] in [3, 4]:
                    Countvalue, BinEdge    = np.histogram(distance[(Countsum  == 7) & (Countsub == 1)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    rho34                  = self.Type[self.Type != self.ParticleType[n][i]][2] - 1
                    particlegr[i, n, :,12] = Countvalue / Nideal / self.rhotype[rho34] #34
                    usedrho[i, n, 12]      = self.rhotype[rho34]

                if self.ParticleType[n][i] in [3, 5]:
                    Countvalue, BinEdge    = np.histogram(distance[(Countsum  == 8) & (Countsub == 2)], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    medium                 = self.Type[self.Type != self.ParticleType[n][i]][2:]
                    rho35                  = medium[medium != 4] - 1
                    particlegr[i, n, :,13] = Countvalue / Nideal / self.rhotype[rho35] #35
                    usedrho[i, n, 13]      = self.rhotype[rho35]

                if self.ParticleType[n][i] in [4, 5]:
                    Countvalue, BinEdge    = np.histogram(distance[Countsum  == 9], bins = MAXBIN, range = (0, MAXBIN * rdelta))
                    rho45                  = self.Type[self.Type != self.ParticleType[n][i]][-1] - 1
                    particlegr[i, n, :,14] = Countvalue / Nideal / self.rhotype[rho45] #45
                    usedrho[i, n, 14]      = self.rhotype[rho45]

        binright -= 0.5 * rdelta #middle of each bin
        for n in range(self.SnapshotNumber - avetime):
            for i in range(self.ParticleNumber):
                aveparticlegr       = particlegr[i, n: n + avetime + 1].mean(axis = 0)
                integralgr          = (aveparticlegr * np.log(aveparticlegr + 1e-12) - (aveparticlegr - 1)) * usedrho[i, n][np.newaxis, :]
                integralgr          = integralgr[:, np.any(aveparticlegr, axis = 0)]   #remove zero columns in gr 
                S2results[i, n]     =-0.5 * np.sum(Areafac(self.ndim) * np.pi * binright**(self.ndim - 1) * integralgr.sum(axis = 1) * rdelta)

        return S2results