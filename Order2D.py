#!/usr/bin/python
# coding = utf-8
#This module is part of an analysis package

Authorinfo = """
             ------------------Name: Yuan-Chao Hu--------------
             --------------Email: ychu0213@gmail.com-----------
             ----------Web: https://yuanchaohu.github.io/------
             """

Docstr = """
         This module calculates bond orientational order at 2D

         The code accounts for both orthogonal and triclinic cells
         """

import os
import numpy  as np 
import pandas as pd 
from   dump   import readdump
from   math   import pi, sqrt
import cmath
from   ParticleNeighbors import Voropp

class BOO2D:
    """ Compute Bond Orientational Order in two dimension """

    def __init__(self, dumpfile, Neighborfile, edgelengthfile='', filetype = 'lammps', moltypes = '', *arg):
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

        edgelengthfile: filename of voronoi cell bonds lengths for each particle
        w.r.t its neighbors, like the face area in 3D. This can be used to weight
        the order parameter; provide False if do not want to weight. (default)
        """
        
        self.dumpfile = dumpfile
        self.filetype = filetype
        self.moltypes = moltypes
        d = readdump(self.dumpfile, 2, self.filetype, self.moltypes) #only at two dimension
        d.read_onefile()
        self.Neighborfile = Neighborfile
        self.edgelengthfile = edgelengthfile

        if len(d.TimeStep) > 1:
            self.TimeStep     = d.TimeStep[1] - d.TimeStep[0]
            if self.TimeStep != d.TimeStep[-1] - d.TimeStep[-2]:
                print ('Warning: *********** dump interval changes **************') 
        self.ParticleNumber     = d.ParticleNumber[0] 
        if d.ParticleNumber[0] != d.ParticleNumber[-1]:
            print ('Warning: ************* Paticle Number Changes **************')
        self.ParticleType   = d.ParticleType
        self.Positions      = np.array(d.Positions)
        self.SnapshotNumber = d.SnapshotNumber
        self.Boxlength      = d.Boxlength[0]
        if not (d.Boxlength[0] == d.Boxlength[-1]).all():
            print ('Warning: *********Box Length Changed from Dump***********')
        self.rhototal   = self.ParticleNumber / np.prod(self.Boxlength)
        self.hmatrix    = d.hmatrix
        self.typecounts = np.unique(self.ParticleType[0], return_counts = True) 
        self.Type       = self.typecounts[0]
        self.TypeNumber = self.typecounts[1]
        print ('Particle Type:', self.Type)
        print ('Particle TypeNumber:', self.TypeNumber)
        if np.sum(self.TypeNumber) != self.ParticleNumber:
            print ('Warning: ****** Sum of Indivdual Types is Not the Total Amount*******')
        
        if self.edgelengthfile:
            print ('----Bond orientational order will be weighted by Voronoi bond lengths----')
        else:
            print ('----Non-weighting is using----')

    def lthorder(self, l = 6, ppp = [1, 1]):
        """ Calculate l-th order in 2D, such as hexatic order

            l is the order ranging from 4 to 8 normally in 2D
            ppp is periodic boundary conditions. 1 for yes and 0 for no
        """

        fneighbor = open(self.Neighborfile, 'r')
        if self.edgelengthfile: 
            fbondlength = open(self.edgelengthfile)
        
        results = np.zeros((self.SnapshotNumber, self.ParticleNumber), dtype = np.complex128)
        for n in range(self.SnapshotNumber):
            hmatrixinv   = np.linalg.inv(self.hmatrix[n])
            Neighborlist = Voropp(fneighbor, self.ParticleNumber) #neighbor list [number, list...]
            if not self.edgelengthfile:
                for i in range(self.ParticleNumber):
                    RIJ = self.Positions[n, Neighborlist[i, 1: Neighborlist[i, 0] + 1]] - self.Positions[n, i]
                    matrixij = np.dot(RIJ, hmatrixinv)
                    RIJ      = np.dot(matrixij - np.rint(matrixij) * ppp, self.hmatrix[n]) #remove PBC
                    theta    = np.arctan2(RIJ[:, 1], RIJ[:, 0])
                    results[n, i] = (np.exp(1j * l * theta)).mean()
            else:
                bondlengthlist = Voropp(fbondlength, self.ParticleNumber)
                for i in range(self.ParticleNumber):
                    RIJ = self.Positions[n, Neighborlist[i, 1: Neighborlist[i, 0] + 1]] - self.Positions[n, i]
                    matrixij = np.dot(RIJ, hmatrixinv)
                    RIJ      = np.dot(matrixij - np.rint(matrixij) * ppp, self.hmatrix[n]) #remove PBC
                    theta    = np.arctan2(RIJ[:, 1], RIJ[:, 0])
                    weights  = bondlengthlist[i, 1:Neighborlist[i, 0] + 1] + 1.0
                    weights /= weights.sum()
                    results[n, i] = (weights*np.exp(1j * l * theta)).sum()

        fneighbor.close()
        if self.edgelengthfile:
            fbondlength.close()
        return results #complex number in array

    def tavephi(self, outputphi = '', outputavephi = '', avet = 0, l = 6, ppp = [1, 1], dt = 0.002):
        """ Compute PHI value and Time Averaged PHI

            outputphi is the outputfile of absolute values of phi
            outputavephi is the outputfile of time averaged phi
            !!!Give outputphi and outputavephi names to calculate the wanted parameters
            l is the order ranging from 4 to 8 normally in 2D
            ppp is periodic boundary conditions. 1 for yes and 0 for no
            avet is the time used to average (Time Unit)
            dt is time step of simulation
        """

        results = np.abs(self.lthorder(l, ppp))

        if avet==0:
            #compute absolute phi
            ParticlePhi = np.column_stack((np.arange(self.ParticleNumber) + 1, results.T))
            if outputphi:
                names = 'id   phil=' + str(l)
                numformat = '%d ' + '%.6f ' * (len(ParticlePhi[0]) - 1)
                np.savetxt(outputphi, ParticlePhi, fmt= numformat, header = names, comments = '')
            print('---------Compute Particle Level PHI Over------------')
            return ParticlePhi

        else:
            #compute time averaged phi
            avet = int(avet / dt / self.TimeStep)
            averesults = np.zeros((self.SnapshotNumber - avet, self.ParticleNumber))
            for n in range(self.SnapshotNumber - avet):
                averesults[n] = results[n: n + avet].mean(axis = 0)

            averesults = np.column_stack((np.arange(self.ParticleNumber) + 1, averesults.T))
            if outputavephi:
                names   = 'id   ave_phil=' + str(l)
                numformat = '%d ' + '%.6f ' * (len(averesults[0]) - 1) 
                np.savetxt(outputavephi, averesults, fmt = numformat, header = names, comments = '')
            print ('---------Compute Particle Level PHI Over------------')
            return averesults

    def spatialcorr(self, l = 6, ppp = [1, 1], rdelta = 0.01, outputfile = ''):
        """ Calculate spatial correlation of bond orientational order
            
            l is the order ranging from 4 to 8 normally in 2D
            ppp is periodic boundary conditions. 1 for yes and 0 for no
            rdelta is the bin size in g(r)
        """

        ParticlePhi = self.lthorder(l, ppp)
        MAXBIN      = int(self.Boxlength.min() / 2.0 / rdelta)
        grresults   = np.zeros((MAXBIN, 3))
        names       = 'r   g(r)   gl(r)   gl/g(r)l=' + str(l)

        for n in range(self.SnapshotNumber):
            hmatrixinv = np.linalg.inv(self.hmatrix[n])
            for i in range(self.ParticleNumber - 1):
                RIJ = self.Positions[n, i + 1:] - self.Positions[n, i]
                #periodic = np.where(np.abs(RIJ / self.Boxlength[np.newaxis, :]) > 0.5, np.sign(RIJ), 0).astype(np.int)
                #RIJ -= self.Boxlength * periodic * ppp #remove PBC
                matrixij = np.dot(RIJ, hmatrixinv)
                RIJ      = np.dot(matrixij - np.rint(matrixij) * ppp, self.hmatrix[n]) #remove PBC
                distance = np.sqrt(np.square(RIJ).sum(axis = 1))
                Countvalue, BinEdge = np.histogram(distance, bins = MAXBIN, range = (0, MAXBIN * rdelta))
                grresults[:, 0] += Countvalue
                PHIIJ = np.real(ParticlePhi[n, i + 1:] * np.conj(ParticlePhi[n, i])) 
                Countvalue, BinEdge = np.histogram(distance, bins = MAXBIN, range = (0, MAXBIN * rdelta), weights = PHIIJ)
                grresults[:, 1] += Countvalue

        binleft  = BinEdge[:-1]
        binright = BinEdge[1:] 
        Nideal   = np.pi * (binright**2 - binleft**2) * self.rhototal
        grresults[:, 0] = grresults[:, 0] * 2 / self.ParticleNumber / self.SnapshotNumber / Nideal
        grresults[:, 1] = grresults[:, 1] * 2 / self.ParticleNumber / self.SnapshotNumber / Nideal
        grresults[:, 2] = np.where(grresults[:, 0] != 0, grresults[:, 1] / grresults[:, 0], np.nan)

        binright = binright - 0.5 * rdelta
        results = np.column_stack((binright, grresults))
        if outputfile:
            np.savetxt(outputfile, results, fmt = '%.6f', header = names, comments = '')

        print ('-----------Get gl(r) results Done-----------')
        return results, names

    def timecorr(self, l = 6, ppp = [1, 1], dt = 0.002, outputfile = ''):
        """ Calculate time correlation of bond orientational order
            
            l is the order ranging from 4 to 8 normally in 2D
            ppp is periodic boundary conditions. 1 for yes and 0 for no
            dt is the time step in simulation
        """

        ParticlePhi = self.lthorder(l, ppp)
        results     = np.zeros((self.SnapshotNumber - 1, 2))
        names       = 't   timecorr_phil=' + str(l)

        cal_phi = pd.DataFrame(np.zeros((self.SnapshotNumber-1))[np.newaxis, :])
        fac_phi = pd.DataFrame(np.zeros((self.SnapshotNumber-1))[np.newaxis, :])
        deltat  = np.zeros(((self.SnapshotNumber - 1), 2), dtype = np.int) #deltat, deltatcounts
        for n in range(self.SnapshotNumber - 1):
            CIJ     = (np.real(ParticlePhi[n + 1:] * np.conj(ParticlePhi[n]))).sum(axis = 1)
            cal_phi = pd.concat([cal_phi, pd.DataFrame(CIJ[np.newaxis, :])])
            CII     = np.repeat((np.square(np.abs(ParticlePhi[n]))).sum(), len(CIJ))
            fac_phi = pd.concat([fac_phi, pd.DataFrame(CII[np.newaxis, :])])

        cal_phi = cal_phi.iloc[1:]
        fac_phi = fac_phi.iloc[1:]
        deltat[:, 0] = np.array(cal_phi.columns) + 1 #time interval
        deltat[:, 1] = np.array(cal_phi.count())     #time interval frequency

        results[:, 0] = deltat[:, 0] * self.TimeStep * dt 
        results[:, 1] = cal_phi.mean() / fac_phi.mean()

        if outputfile:
            np.savetxt(outputfile, results, fmt='%.6f', header = names, comments = '')

        print ('-----------------Compute time correlation of phi Over--------------------')
        return results, names