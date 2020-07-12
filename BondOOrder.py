#!/usr/bin/python
# coding = utf-8
#This module is part of an analysis package

Authorinfo = """
             ------------------Name: Yuan-Chao Hu--------------
             --------------Email: ychu0213@gmail.com-----------
             ----------Web: https://yuanchaohu.github.io/------
             """

Docstr = """
         This module calculates bond orientational order in 3D
         Including traditional quantatities and area-weighted ones
         Calculate from q (Q) to sij to Gl(r) to w (W) and w_cap (W_cap)
         Calculate both time correlation and spatial correlation

         This code accounts for both orthogonal and triclinic cells
         """

import os
import numpy  as np 
import pandas as pd 
from   dump   import readdump
from   math   import pi, sqrt
import SphericalHarmonics 
from   ParticleNeighbors    import Voropp
from   sympy.physics.wigner import wigner_3j

def SPfunction(l, theta, phi):
    """ Choose Spherical Harmonics Using l """
    if l == 2:
        return SphericalHarmonics.SphHarm2(theta, phi)
    if l == 3:
        return SphericalHarmonics.SphHarm3(theta, phi)
    if l == 4:
        return SphericalHarmonics.SphHarm4(theta, phi)
    if l == 5:
        return SphericalHarmonics.SphHarm5(theta, phi)
    if l == 6:
        return SphericalHarmonics.SphHarm6(theta, phi)
    if l == 7:
        return SphericalHarmonics.SphHarm7(theta, phi)
    if l == 8:
        return SphericalHarmonics.SphHarm8(theta, phi)
    if l == 9:
        return SphericalHarmonics.SphHarm9(theta, phi)
    if l == 10:
        return SphericalHarmonics.SphHarm10(theta, phi)
    if l > 10:
        return SphericalHarmonics.SphHarm_above(l, theta, phi)

def Wignerindex(l):
    """ Define Wigner 3-j symbol """
    selected = []
    for m1 in range(-l, l + 1):
        for m2 in range(-l, l + 1):
            for m3 in range(-l, l + 1):
                if m1 + m2 + m3 ==0:
                    windex = wigner_3j(l, l, l, m1, m2, m3).evalf()
                    selected.append(np.array([m1, m2, m3, windex]))

    return np.ravel(np.array(selected)).reshape(-1, 4)

class BOO3D:
    """ Compute Bond Orientational Order in three dimension """

    def __init__(self, dumpfile, Neighborfile, faceareafile = '', filetype = 'lammps', moltypes = '', *arg):
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
        self.dumpfile = dumpfile
        self.filetype = filetype
        self.moltypes = moltypes
        d = readdump(self.dumpfile, 3, self.filetype, self.moltypes)
        d.read_onefile()
        self.Neighborfile = Neighborfile
        self.faceareafile = faceareafile

        if len(d.TimeStep) > 1:
            self.TimeStep = d.TimeStep[1] - d.TimeStep[0]
            if self.TimeStep != d.TimeStep[-1] - d.TimeStep[-2]:
                print ('Warning: *********** dump interval changes **************') 
        self.ParticleNumber     = d.ParticleNumber[0] 
        if d.ParticleNumber[0] != d.ParticleNumber[-1]:
            raise ValueError('************* Paticle Number Changes **************')
        self.ParticleType   = d.ParticleType
        self.Positions      = d.Positions
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

    def qlmQlm(self, l, ppp, AreaR):
        """ BOO of the l-fold symmetry as a 2l + 1 vector

            AreaR = 0 indicates calculate traditional qlm and Qlm
            AreaR = 1 indicates calculate voronoi polyhedron face area weighted qlm and Qlm
        """

        fneighbor = open(self.Neighborfile, 'r')
        if AreaR == 1: ffacearea = open(self.faceareafile, 'r')

        smallqlm = []
        largeQlm = []
        for n in range(self.SnapshotNumber):
            hmatrixinv = np.linalg.inv(self.hmatrix[n])
            if AreaR == 0:   #calculate traditional qlm and Qlm
                Neighborlist     = Voropp(fneighbor, self.ParticleNumber)  #neighbor list [number, list....]
                Particlesmallqlm = np.zeros((self.ParticleNumber, 2 * l + 1), dtype = np.complex128)
                for i in range(self.ParticleNumber):
                    RIJ = self.Positions[n][Neighborlist[i, 1: (Neighborlist[i, 0] + 1)]] - self.Positions[n][i]
                    #periodic = np.where(np.abs(RIJ / self.Boxlength[np.newaxis, :]) > 0.50, np.sign(RIJ), 0).astype(np.int)
                    #RIJ -= self.Boxlength * periodic * ppp    #remove Periodic boundary conditions
                    matrixij = np.dot(RIJ, hmatrixinv)
                    RIJ      = np.dot(matrixij - np.rint(matrixij) * ppp, self.hmatrix[n]) #remove PBC
                    theta = np.arccos(RIJ[:, 2] / np.sqrt(np.square(RIJ).sum(axis = 1)))
                    phi   = np.arctan2(RIJ[:, 1], RIJ[:, 0])
                    for j in range(Neighborlist[i, 0]):
                        Particlesmallqlm[i] += SPfunction(l, theta[j], phi[j]) #-l ... 0 ... l
                Particlesmallqlm = Particlesmallqlm / (Neighborlist[:, 0])[:, np.newaxis]
                smallqlm.append(Particlesmallqlm)

            elif AreaR == 1: #calculate voronoi polyhedron facearea weighted qlm and Qlm 
                Neighborlist = Voropp(fneighbor, self.ParticleNumber)  #neighbor list [number, list....]
                facearealist = Voropp(ffacearea, self.ParticleNumber)  #facearea list [number, list....]
                facearealist[:, 1:] = np.where(facearealist[:, 1:] != 0, facearealist[:, 1:] + 1, facearealist[:, 1:]) #becase -1 has been added in Voropp()
                faceareafrac = facearealist[:, 1:] / facearealist[:, 1:].sum(axis = 1)[:, np.newaxis] #facearea fraction
                Particlesmallqlm = np.zeros((self.ParticleNumber, 2 * l + 1), dtype = np.complex128)
                for i in range(self.ParticleNumber):
                    RIJ = self.Positions[n][Neighborlist[i, 1: (Neighborlist[i, 0] + 1)]] - self.Positions[n][i]
                    #periodic = np.where(np.abs(RIJ / self.Boxlength[np.newaxis, :]) > 0.50, np.sign(RIJ), 0).astype(np.int)
                    #RIJ -= self.Boxlength * periodic * ppp    #remove PBC
                    matrixij = np.dot(RIJ, hmatrixinv)
                    RIJ      = np.dot(matrixij - np.rint(matrixij) * ppp, self.hmatrix[n]) #remove PBC
                    theta = np.arccos(RIJ[:, 2] / np.sqrt(np.square(RIJ).sum(axis = 1)))
                    phi   = np.arctan2(RIJ[:, 1], RIJ[:, 0])
                    for j in range(Neighborlist[i, 0]):
                        Particlesmallqlm[i] += np.array(SPfunction(l, theta[j], phi[j])) * faceareafrac[i, j] #-l ... 0 ... l
                smallqlm.append(Particlesmallqlm)


            ParticlelargeQlm = np.copy(Particlesmallqlm)  ####must use copy otherwise only rename it 
            for i in range(self.ParticleNumber):
                for j in range(Neighborlist[i, 0]):
                    ParticlelargeQlm[i] += Particlesmallqlm[Neighborlist[i, j+1]]
            ParticlelargeQlm = ParticlelargeQlm / (1 + Neighborlist[:, 0])[:, np.newaxis]
            largeQlm.append(ParticlelargeQlm)
        
        fneighbor.close()
        if AreaR == 1: ffacearea.close()
        return (smallqlm, largeQlm)  #complex number 

    def qlQl(self, l, ppp = [1,1,1], AreaR = 0, outputql = '', outputQl = ''):
        """ Calculate BOO ql and Ql (coarse-grained by the first neighbor shell)

            AreaR = 0 indicates calculate traditional ql and Ql
            AreaR = 1 indicates calculate voronoi polyhedron face area weighted ql and Ql
            Give names to outputql and outputQl to store the data
        """
        print ('----Calculate the rotational invariants ql & Ql----')

        (smallqlm, largeQlm) = self.qlmQlm(l, ppp, AreaR)
        smallql = np.sqrt(4 * pi / (2 * l + 1) * np.square(np.abs(smallqlm)).sum(axis = 2))
        smallql = np.column_stack((np.arange(self.ParticleNumber) + 1, smallql.T))
        if outputql:
            names = 'id  ql  l=' + str(l)
            numformat = '%d ' + '%.6f ' * (len(smallql[0]) - 1) 
            np.savetxt(outputql, smallql, fmt=numformat, header = names, comments = '')

        largeQl = np.sqrt(4 * pi / (2 * l + 1) * np.square(np.abs(largeQlm)).sum(axis = 2))
        largeQl = np.column_stack((np.arange(self.ParticleNumber) + 1, largeQl.T))
        if outputQl:
            names = 'id  Ql  l=' + str(l)
            numformat = '%d ' + '%.6f ' * (len(largeQl[0]) - 1) 
            np.savetxt(outputQl, largeQl, fmt=numformat, header = names, comments = '')
        
        print ('-------------Calculate ql and Ql over-----------')
        return (smallql, largeQl)

    def sijsmallql(self, l, ppp = [1,1,1], AreaR = 0, c = 0.7, outputql = '', outputsij = ''):
        """ Calculate Crystal Nuclei Criterion s(i, j) based on qlm  

            AreaR = 0 indicates calculate s(i, j) based on traditional qlm 
            AreaR = 1 indicates calculate s(i, j) based on voronoi polyhedron face area weighted qlm
            c is a cutoff demonstrating whether a bond is crystalline or not
            Give names to outputql and outputsij to store the results
        """
        print ('---- Calculate Crystal Nuclei Criterion s(i, j) based on ql ----')

        MaxNeighbor = 100 #the considered maximum number of neighbors
        (smallqlm, largeQlm) = self.qlmQlm(l, ppp, AreaR)
        fneighbor = open(self.Neighborfile, 'r')
        results = np.zeros((1, 3))
        resultssij = np.zeros((1,  MaxNeighbor + 1))
        for n in range(self.SnapshotNumber):
            Neighborlist = Voropp(fneighbor, self.ParticleNumber)  #neighbor list [number, list....]
            sij = np.zeros((self.ParticleNumber, MaxNeighbor))
            sijresults = np.zeros((self.ParticleNumber, 3))
            if (Neighborlist[:, 0] > MaxNeighbor).any(): print ('********Warning: Too Many Neighbors*********')
            for i in range(self.ParticleNumber):
                for j in range(Neighborlist[i, 0]):
                    sijup = (smallqlm[n][i] * np.conj(smallqlm[n][Neighborlist[i, j+1]])).sum()
                    sijdown = np.sqrt(np.square(np.abs(smallqlm[n][i])).sum()) * np.sqrt(np.square(np.abs(smallqlm[n][Neighborlist[i, j+1]])).sum())
                    sij[i, j] = np.real(sijup / sijdown)
            sijresults[:, 0] = np.arange(self.ParticleNumber) + 1 #particle id
            sijresults[:, 1] = (np.where(sij > c, 1, 0)).sum(axis = 1)  #bond number 
            sijresults[:, 2] = np.where(sijresults[:, 1] > Neighborlist[:, 0] / 2, 1, 0) #crystalline
            results = np.vstack((results, sijresults))
            resultssij = np.vstack((resultssij, np.column_stack((sijresults[:, 0] ,sij))))

        if outputql:
            names = 'id  sijcrystalbondnum  crystalline.l=' + str(l)
            np.savetxt(outputql, results[1:], fmt='%d', header = names, comments = '')

        if outputsij:
            names = 'id s(i, j)  l=' + str(l)
            formatsij = '%d ' + '%.6f ' * MaxNeighbor
            np.savetxt(outputsij, resultssij[1:], fmt=formatsij, header = names, comments = '')

        fneighbor.close()
        print ('-------------Calculate s(i, j) based on ql over-----------')
        return resultssij[1:] #individual value of sij

    def sijlargeQl(self, l, ppp = [1,1,1], AreaR = 0, c = 0.7, outputQl = '', outputsij = ''):
        """ Calculate Crystal Nuclei Criterion s(i, j) based on Qlm  

            AreaR = 0 indicates calculate s(i, j) based on traditional Qlm
            AreaR = 1 indicates calculate s(i, j) based on voronoi polyhedron face area weighted Qlm
            c is a cutoff demonstrating whether a bond is crystalline or not
            Give a name to outputQl and outputsij to store the results
        """
        print ('---- Calculate Crystal Nuclei Criterion s(i, j) based on Ql ----')

        MaxNeighbor = 100 #the considered maximum number of neighbors
        (smallqlm, largeQlm) = self.qlmQlm(l, ppp, AreaR)
        fneighbor = open(self.Neighborfile, 'r')
        results = np.zeros((1, 3))
        resultssij = np.zeros((1,  MaxNeighbor + 1))
        for n in range(self.SnapshotNumber):
            Neighborlist = Voropp(fneighbor, self.ParticleNumber)  #neighbor list [number, list....]
            sij = np.zeros((self.ParticleNumber, MaxNeighbor))
            sijresults = np.zeros((self.ParticleNumber, 3))
            if (Neighborlist[:, 0] > MaxNeighbor).any(): print ('********Warning: Too Many Neighbors*********')
            for i in range(self.ParticleNumber):
                for j in range(Neighborlist[i, 0]):
                    sijup = (largeQlm[n][i] * np.conj(largeQlm[n][Neighborlist[i, j+1]])).sum()
                    sijdown = np.sqrt(np.square(np.abs(largeQlm[n][i])).sum()) * np.sqrt(np.square(np.abs(largeQlm[n][Neighborlist[i, j+1]])).sum())
                    sij[i, j] = np.real(sijup / sijdown)
            sijresults[:, 0] = np.arange(self.ParticleNumber) + 1 #particle id
            sijresults[:, 1] = (np.where(sij > c, 1, 0)).sum(axis = 1)  #bond number 
            sijresults[:, 2] = np.where(sijresults[:, 1] > Neighborlist[:, 0] / 2, 1, 0) #crystalline
            results = np.vstack((results, sijresults))
            resultssij = np.vstack((resultssij, np.column_stack((sijresults[:, 0] ,sij))))

        if outputQl:
            names = 'id  sijcrystalbondnum  crystalline.l=' + str(l)
            np.savetxt(outputQl, results[1:], fmt='%d', header = names, comments = '')
            
        if outputsij:
            names = 'id  s(i, j)  l=' + str(l)
            formatsij = '%d ' + '%.6f ' * MaxNeighbor
            np.savetxt(outputsij, resultssij[1:], fmt=formatsij, header = names, comments = '')

        fneighbor.close()
        print ('-------------Calculate s(i, j) based on Ql over-----------')
        return resultssij[1:] #individual value of sij

    def GllargeQ(self, l, ppp = [1,1,1], rdelta = 0.01, AreaR = 0, outputgl = ''):
        """ Calculate bond order spatial correlation function Gl(r) based on Qlm

            AreaR = 0 indicates calculate Gl(r) based on traditional Qlm
            AreaR = 1 indicates calculate s(i, j) based on voronoi polyhedron face area weighted Qlm
            rdelta is the bin size in calculating g(r) and Gl(r)
        """
        print ('---- Calculate bond order correlation Gl based on Ql ----')

        (smallqlm, largeQlm) = self.qlmQlm(l, ppp, AreaR)
        MAXBIN     = int(self.Boxlength.min() / 2.0 / rdelta)
        grresults  = np.zeros((MAXBIN, 3))
        for n in range(self.SnapshotNumber):
            hmatrixinv = np.linalg.inv(self.hmatrix[n])
            for i in range(self.ParticleNumber - 1):
                RIJ      = self.Positions[n][i+1:] - self.Positions[n][i]
                #periodic = np.where(np.abs(RIJ / self.Boxlength[np.newaxis, :]) > 0.50, np.sign(RIJ), 0).astype(np.int)
                #RIJ -= self.Boxlength * periodic * ppp    #remove PBC
                matrixij = np.dot(RIJ, hmatrixinv)
                RIJ      = np.dot(matrixij - np.rint(matrixij) * ppp, self.hmatrix[n]) #remove PBC
                distance = np.sqrt(np.square(RIJ).sum(axis = 1))
                Countvalue, BinEdge = np.histogram(distance, bins = MAXBIN, range = (0, MAXBIN * rdelta))
                grresults[:, 0] += Countvalue
                QIJ      = np.real((largeQlm[n][i + 1:] * np.conj(largeQlm[n][i])).sum(axis = 1))
                Countvalue, BinEdge = np.histogram(distance, bins = MAXBIN, range = (0, MAXBIN * rdelta), weights = QIJ)
                grresults[:, 1] += Countvalue

        binleft    = BinEdge[:-1]   #real value of each bin edge, not index
        binright   = BinEdge[1:]   #len(Countvalue) = len(BinEdge) - 1
        Nideal     = 4/3.0 * np.pi * (binright**3 - binleft**3) * self.rhototal 
        grresults[:, 0]  = grresults[:, 0] * 2 / self.ParticleNumber / self.SnapshotNumber / Nideal
        grresults[:, 1]  = (4 * pi / (2 * l + 1)) * grresults[:, 1] * 2 / self.ParticleNumber / self. SnapshotNumber / Nideal
        grresults[:, 2]  = np.where(grresults[:, 0] != 0, grresults[:, 1] / grresults[:, 0], np.nan)

        binright = binright - 0.5 * rdelta #middle of each bin
        results  = np.column_stack((binright, grresults))
        names    = 'r  g(r)  Gl(r)  Gl/gl=' + str(l)
        if outputgl:
            np.savetxt(outputgl, results, fmt='%.6f', header = names, comments = '')

        print ('---------- Get Gl(r) results based on Ql over ---------')
        return results, names

    def Glsmallq(self, l, ppp = [1,1,1], rdelta = 0.01, AreaR = 0, outputgl = ''):
        """ Calculate bond order spatial correlation function Gl(r) based on qlm

            AreaR = 0 indicates calculate Gl(r) based on traditional qlm
            AreaR = 1 indicates calculate Gl(r) based on voronoi polyhedron face area weighted qlm
            rdelta is the bin size in calculating g(r) and Gl(r)
        """
        print ('---- Calculate bond order correlation Gl based on ql ----')

        (smallqlm, largeQlm) = self.qlmQlm(l, ppp, AreaR)
        MAXBIN     = int(self.Boxlength.min() / 2.0 / rdelta)
        grresults  = np.zeros((MAXBIN, 3))
        for n in range(self.SnapshotNumber):
            hmatrixinv = np.linalg.inv(self.hmatrix[n])
            for i in range(self.ParticleNumber - 1):
                RIJ      = self.Positions[n][i+1:] - self.Positions[n][i]
                #periodic = np.where(np.abs(RIJ / self.Boxlength[np.newaxis, :]) > 0.50, np.sign(RIJ), 0).astype(np.int)
                #RIJ -= self.Boxlength * periodic * ppp    #remove PBC
                matrixij = np.dot(RIJ, hmatrixinv)
                RIJ      = np.dot(matrixij - np.rint(matrixij) * ppp, self.hmatrix[n]) #remove PBC
                distance = np.sqrt(np.square(RIJ).sum(axis = 1))
                Countvalue, BinEdge = np.histogram(distance, bins = MAXBIN, range = (0, MAXBIN * rdelta))
                grresults[:, 0] += Countvalue
                QIJ      = np.real((smallqlm[n][i + 1:] * np.conj(smallqlm[n][i])).sum(axis = 1))
                Countvalue, BinEdge = np.histogram(distance, bins = MAXBIN, range = (0, MAXBIN * rdelta), weights = QIJ)
                grresults[:, 1] += Countvalue

        binleft    = BinEdge[:-1]   #real value of each bin edge, not index
        binright   = BinEdge[1:]   #len(Countvalue) = len(BinEdge) - 1
        Nideal     = 4/3.0 * np.pi * (binright**3 - binleft**3) * self.rhototal 
        grresults[:, 0]  = grresults[:, 0] * 2 / self.ParticleNumber / self.SnapshotNumber / Nideal
        grresults[:, 1]  = (4 * pi / (2 * l + 1)) * grresults[:, 1] * 2 / self.ParticleNumber / self. SnapshotNumber / Nideal
        grresults[:, 2]  = np.where(grresults[:, 0] != 0, grresults[:, 1] / grresults[:, 0], np.nan)

        binright = binright - 0.5 * rdelta #middle of each bin
        results  = np.column_stack((binright, grresults))
        names    = 'r  g(r)  Gl(r)  Gl/gl=' + str(l)
        if outputgl:
            np.savetxt(outputgl, results, fmt='%.6f', header = names, comments = '')

        print ('---------- Get Gl(r) results based on ql over ---------')
        return results, names


    def smallwcap(self, l, ppp = [1,1,1], AreaR = 0, outputw = '', outputwcap = ''):
        """ Calculate wigner 3-j symbol boo based on qlm

            AreaR = 0 indicates calculation based on traditional qlm
            AreaR = 1 indicates calculation based on voronoi polyhedron face area weighted qlm
            Give names to outputw, outputwcap to store the data
        """
        print ('---- Calculate bond Orientational order w (normalized) based on qlm ----')

        (smallqlm, largeQlm) = self.qlmQlm(l, ppp, AreaR)
        smallqlm = np.array(smallqlm)
        smallw = np.zeros((self.SnapshotNumber, self.ParticleNumber))
        Windex = Wignerindex(l)
        w3j    = Windex[:, 3]
        Windex = Windex[:, :3].astype(np.int) + l 
        for n in range(self.SnapshotNumber):
            for i in range(self.ParticleNumber):
                smallw[n, i] = (np.real(np.prod(smallqlm[n, i, Windex], axis = 1)) * w3j).sum()
       
        smallw = np.column_stack((np.arange(self.ParticleNumber) + 1, smallw.T))
        if outputw:
            names = 'id  wl  l=' + str(l)
            numformat = '%d ' + '%.10f ' * (len(smallw[0]) - 1) 
            np.savetxt(outputw, smallw, fmt=numformat, header = names, comments = '')
   
        smallwcap = np.power(np.square(np.abs(np.array(smallqlm))).sum(axis = 2), -3 / 2).T * smallw[:, 1:]
        smallwcap = np.column_stack((np.arange(self.ParticleNumber) + 1, smallwcap))
        if outputwcap:
            names = 'id  wlcap  l=' + str(l)
            numformat = '%d ' + '%.8f ' * (len(smallwcap[0]) - 1) 
            np.savetxt(outputwcap, smallwcap, fmt=numformat, header = names, comments = '')
        
        print ('------------- Calculate BOO w and normalized (cap) one over ----------------')
        return (smallw, smallwcap)


    def largeWcap(self, l, ppp = [1,1,1], AreaR = 0, outputW = '', outputWcap = ''):
        """ Calculate wigner 3-j symbol boo based on Qlm

            AreaR = 0 indicates calculation based on traditional Qlm
            AreaR = 1 indicates calculation based on voronoi polyhedron face area weighted Qlm
            Give names to outputW, outputWcap to store the data
        """
        print ('---- Calculate bond Orientational order W (normalized) based on Qlm ----')

        (smallqlm, largeQlm) = self.qlmQlm(l, ppp, AreaR)
        largeQlm = np.array(largeQlm)
        largew = np.zeros((self.SnapshotNumber, self.ParticleNumber))
        Windex = Wignerindex(l)
        w3j    = Windex[:, 3]
        Windex = Windex[:, :3].astype(np.int) + l 
        for n in range(self.SnapshotNumber):
            for i in range(self.ParticleNumber):
                largew[n, i] = (np.real(np.prod(largeQlm[n, i, Windex], axis = 1)) * w3j).sum()
       
        largew = np.column_stack((np.arange(self.ParticleNumber) + 1, np.real(largew.T)))
        if outputW:
            names = 'id  Wl  l=' + str(l)
            numformat = '%d ' + '%.10f ' * (len(largew[0]) - 1) 
            np.savetxt(outputW, largew, fmt=numformat, header = names, comments = '')
   
        largewcap = np.power(np.square(np.abs(np.array(largeQlm))).sum(axis = 2), -3 / 2).T * largew[:, 1:]
        largewcap = np.column_stack((np.arange(self.ParticleNumber) + 1, largewcap))
        if outputWcap:
            names = 'id  Wlcap  l=' + str(l)
            numformat = '%d ' + '%.8f ' * (len(largewcap[0]) - 1) 
            np.savetxt(outputWcap, largewcap, fmt=numformat, header = names, comments = '')
        
        print ('------------- Calculate BOO W and normalized (cap) one over ----------------')
        return (largew, largewcap)


    def timecorr(self, l, ppp = [1,1,1], AreaR = 0, dt = 0.002, outputfile = ''):
        """ Calculate time correlation of qlm and Qlm

            AreaR = 0 indicates calculate traditional ql and Ql
            AreaR = 1 indicates calculate voronoi polyhedron face area weighted ql and Ql
        """
        print ('----Calculate the time correlation of qlm & Qlm----')

        (smallqlm, largeQlm) = self.qlmQlm(l, ppp, AreaR)
        smallqlm = np.array(smallqlm)
        largeQlm = np.array(largeQlm)
        results = np.zeros((self.SnapshotNumber - 1, 3))
        names = 't   timecorr_q   timecorr_Ql=' + str(l)

        cal_smallq = pd.DataFrame(np.zeros((self.SnapshotNumber-1))[np.newaxis, :])
        cal_largeQ = pd.DataFrame(np.zeros((self.SnapshotNumber-1))[np.newaxis, :])
        fac_smallq = pd.DataFrame(np.zeros((self.SnapshotNumber-1))[np.newaxis, :])
        fac_largeQ = pd.DataFrame(np.zeros((self.SnapshotNumber-1))[np.newaxis, :])
        deltat     = np.zeros(((self.SnapshotNumber - 1), 2), dtype = np.int) #deltat, deltatcounts
        for n in range(self.SnapshotNumber - 1):
            CIJsmallq  = (np.real((smallqlm[n + 1:] * np.conj(smallqlm[n]))).sum(axis = 2)).sum(axis = 1)
            cal_smallq = pd.concat([cal_smallq, pd.DataFrame(CIJsmallq[np.newaxis, :])])
            CIIsmallq  = np.repeat((np.square(np.abs(smallqlm[n])).sum()), len(CIJsmallq)) #consider initial snapshot
            fac_smallq = pd.concat([fac_smallq, pd.DataFrame(CIIsmallq[np.newaxis, :])])
            CIJlargeQ  = (np.real((largeQlm[n + 1:] * np.conj(largeQlm[n]))).sum(axis = 2)).sum(axis = 1)
            cal_largeQ = pd.concat([cal_largeQ, pd.DataFrame(CIJlargeQ[np.newaxis, :])])
            CIIlargeQ  = np.repeat((np.square(np.abs(largeQlm[n])).sum()), len(CIJlargeQ))
            fac_largeQ = pd.concat([fac_largeQ, pd.DataFrame(CIIlargeQ[np.newaxis, :])])

        cal_smallq = cal_smallq.iloc[1:]
        cal_largeQ = cal_largeQ.iloc[1:]
        fac_smallq = fac_smallq.iloc[1:]
        fac_largeQ = fac_largeQ.iloc[1:]
        deltat[:, 0] = np.array(cal_smallq.columns) + 1 #Timeinterval
        deltat[:, 1] = np.array(cal_smallq.count())     #Timeinterval frequency
       
        results[:, 0] = deltat[:, 0] * self.TimeStep * dt 
        results[:, 1] = cal_smallq.mean() * (4 * pi / (2 * l + 1)) / fac_smallq.mean()
        results[:, 2] = cal_largeQ.mean() * (4 * pi / (2 * l + 1)) / fac_largeQ.mean()

        if outputfile: 
            np.savetxt(outputfile, results, fmt='%.6f', header = names, comments = '')
        
        print ('-----------------Compute time correlation of qlm and Qlm Over--------------------')
        return results

    def cal_multiple(self, l, ppp = [1,1,1], AreaR = 0, c = 0.7, outpath = './', cqlQl = 0, csijsmallql = 0, csijlargeQl = 0, csmallwcap = 0, clargeWcap = 0):
        """Calculate multiple order parameters at the same time"""

        if not os.path.exists(outpath): os.makedirs(outpath)
        namestr = outpath + '.'.join(os.path.basename(self.dumpfile).split('.')[:-1])
        print ('----calculate multiple order parameters together------')
        print ('the common namestr of output filenames is %s'%namestr)
        (smallqlm, largeQlm) = self.qlmQlm(l, ppp, AreaR)

        if cqlQl:
            smallql  = np.sqrt(4 * pi / (2 * l + 1) * np.square(np.abs(smallqlm)).sum(axis = 2))
            smallql  = np.column_stack((np.arange(self.ParticleNumber) + 1, smallql.T))
            outputql = namestr + '.smallq_l%d.dat'%l
            names    = 'id  ql  l=' + str(l)
            numformat = '%d ' + '%.6f ' * (len(smallql[0]) - 1) 
            np.savetxt(outputql, smallql, fmt=numformat, header = names, comments = '')

            largeQl = np.sqrt(4 * pi / (2 * l + 1) * np.square(np.abs(largeQlm)).sum(axis = 2))
            largeQl = np.column_stack((np.arange(self.ParticleNumber) + 1, largeQl.T))
            outputQl = namestr + '.largeQ_l%d.dat'%l
            names = 'id  Ql  l=' + str(l)
            numformat = '%d ' + '%.6f ' * (len(largeQl[0]) - 1) 
            np.savetxt(outputQl, largeQl, fmt=numformat, header = names, comments = '')
            
            print ('-------------Calculate ql and Ql over-----------')

        if csijsmallql:
            MaxNeighbor = 100 #the considered maximum number of neighbors
            fneighbor = open(self.Neighborfile, 'r')
            results = np.zeros((1, 3))
            for n in range(self.SnapshotNumber):
                Neighborlist = Voropp(fneighbor, self.ParticleNumber)  #neighbor list [number, list....]
                sij = np.zeros((self.ParticleNumber, MaxNeighbor))
                sijresults = np.zeros((self.ParticleNumber, 3))
                if (Neighborlist[:, 0] > MaxNeighbor).any(): print ('********Warning: Too Many Neighbors*********')
                for i in range(self.ParticleNumber):
                    for j in range(Neighborlist[i, 0]):
                        sijup = (smallqlm[n][i] * np.conj(smallqlm[n][Neighborlist[i, j+1]])).sum()
                        sijdown = np.sqrt(np.square(np.abs(smallqlm[n][i])).sum()) * np.sqrt(np.square(np.abs(smallqlm[n][Neighborlist[i, j+1]])).sum())
                        sij[i, j] = np.real(sijup / sijdown)
                sijresults[:, 0] = np.arange(self.ParticleNumber) + 1 #particle id
                sijresults[:, 1] = (np.where(sij > c, 1, 0)).sum(axis = 1)  #bond number 
                sijresults[:, 2] = np.where(sijresults[:, 1] > Neighborlist[:, 0] / 2, 1, 0) #crystalline
                results = np.vstack((results, sijresults))

            outputql = namestr + '.sij.smallq_l%d.dat'%l
            names = 'id  sijcrystalbondnum  crystalline.l=' + str(l)
            np.savetxt(outputql, results[1:], fmt='%d', header = names, comments = '')
            fneighbor.close()
            print ('-------------Calculate s(i, j) based on ql over-----------')

        if csijlargeQl:
            MaxNeighbor = 100 #the considered maximum number of neighbors
            fneighbor = open(self.Neighborfile, 'r')
            results = np.zeros((1, 3))
            for n in range(self.SnapshotNumber):
                Neighborlist = Voropp(fneighbor, self.ParticleNumber)  #neighbor list [number, list....]
                sij = np.zeros((self.ParticleNumber, MaxNeighbor))
                sijresults = np.zeros((self.ParticleNumber, 3))
                if (Neighborlist[:, 0] > MaxNeighbor).any(): print ('********Warning: Too Many Neighbors*********')
                for i in range(self.ParticleNumber):
                    for j in range(Neighborlist[i, 0]):
                        sijup = (largeQlm[n][i] * np.conj(largeQlm[n][Neighborlist[i, j+1]])).sum()
                        sijdown = np.sqrt(np.square(np.abs(largeQlm[n][i])).sum()) * np.sqrt(np.square(np.abs(largeQlm[n][Neighborlist[i, j+1]])).sum())
                        sij[i, j] = np.real(sijup / sijdown)
                sijresults[:, 0] = np.arange(self.ParticleNumber) + 1 #particle id
                sijresults[:, 1] = (np.where(sij > c, 1, 0)).sum(axis = 1)  #bond number 
                sijresults[:, 2] = np.where(sijresults[:, 1] > Neighborlist[:, 0] / 2, 1, 0) #crystalline
                results = np.vstack((results, sijresults))

            outputQl = namestr + '.sij.largeQ_l%d.dat'%l
            names = 'id  sijcrystalbondnum  crystalline.l=' + str(l)
            np.savetxt(outputQl, results[1:], fmt='%d', header = names, comments = '')
            fneighbor.close()
            print ('-------------Calculate s(i, j) based on Ql over-----------')

        if csmallwcap:
            smallqlm = np.array(smallqlm)
            smallw = np.zeros((self.SnapshotNumber, self.ParticleNumber))
            Windex = Wignerindex(l)
            w3j    = Windex[:, 3]
            Windex = Windex[:, :3].astype(np.int) + l 
            for n in range(self.SnapshotNumber):
                for i in range(self.ParticleNumber):
                    smallw[n, i] = (np.real(np.prod(smallqlm[n, i, Windex], axis = 1)) * w3j).sum()
           
            smallw = np.column_stack((np.arange(self.ParticleNumber) + 1, smallw.T))
            outputw = namestr + '.smallw_l%d.dat'%l
            names = 'id  wl  l=' + str(l)
            numformat = '%d ' + '%.10f ' * (len(smallw[0]) - 1) 
            np.savetxt(outputw, smallw, fmt=numformat, header = names, comments = '')
       
            smallwcap = np.power(np.square(np.abs(np.array(smallqlm))).sum(axis = 2), -3 / 2).T * smallw[:, 1:]
            smallwcap = np.column_stack((np.arange(self.ParticleNumber) + 1, smallwcap))
            outputwcap = namestr + '.smallwcap_l%d.dat'%l
            names = 'id  wlcap  l=' + str(l)
            numformat = '%d ' + '%.8f ' * (len(smallwcap[0]) - 1) 
            np.savetxt(outputwcap, smallwcap, fmt=numformat, header = names, comments = '')
            print ('------------- Calculate BOO w and normalized (cap) one over ----------------')

        if clargeWcap:
            largeQlm = np.array(largeQlm)
            largew = np.zeros((self.SnapshotNumber, self.ParticleNumber))
            Windex = Wignerindex(l)
            w3j    = Windex[:, 3]
            Windex = Windex[:, :3].astype(np.int) + l 
            for n in range(self.SnapshotNumber):
                for i in range(self.ParticleNumber):
                    largew[n, i] = (np.real(np.prod(largeQlm[n, i, Windex], axis = 1)) * w3j).sum()
           
            largew = np.column_stack((np.arange(self.ParticleNumber) + 1, np.real(largew.T)))
            outputW = namestr + '.largeW_l%d.dat'%l
            names = 'id  Wl  l=' + str(l)
            numformat = '%d ' + '%.10f ' * (len(largew[0]) - 1) 
            np.savetxt(outputW, largew, fmt=numformat, header = names, comments = '')
       
            largewcap = np.power(np.square(np.abs(np.array(largeQlm))).sum(axis = 2), -3 / 2).T * largew[:, 1:]
            largewcap = np.column_stack((np.arange(self.ParticleNumber) + 1, largewcap))
            outputWcap = namestr + '.largeWcap_l%d.dat'%l
            names = 'id  Wlcap  l=' + str(l)
            numformat = '%d ' + '%.8f ' * (len(largewcap[0]) - 1) 
            np.savetxt(outputWcap, largewcap, fmt=numformat, header = names, comments = '')        
            print ('------------- Calculate BOO W and normalized (cap) one over ----------------')

        print ('----calculate multiple order parameters together DONE------')