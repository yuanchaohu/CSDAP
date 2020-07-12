#!/usr/bin/python
# coding = utf-8
#This module is part of an analysis package

Authorinfo = """
             ------------------Name: Yuan-Chao Hu--------------
             --------------Email: ychu0213@gmail.com-----------
             ----------Web: https://yuanchaohu.github.io/------
             """

Docstr = """
         This module is responsible for calculating structure factors
         Then the overall and partial S(q) will be written to a file

         The module accounts for systems ranging from unary to senary
         each function can be called seperately similar to getresults()
         """

import os
import numpy  as np 
import pandas as pd 
from dump import readdump
from math import pi 

def wavevector3d(Numofq = 500):
    """ Define Wave Vector for Structure Factors at three dimension """

    wavenumber = np.square(np.arange(Numofq))
    wavevector = []
    for a in range(Numofq):
        for b in range(Numofq):
            for c in range(Numofq):
                d = a**2 + b**2 + c**2
                if d in wavenumber: 
                    wavevector.append(np.array([d, a, b, c]))
    wavevector = np.ravel(np.array(wavevector))[4:].reshape((-1, 4))
    wavevector = wavevector[wavevector[:, 0].argsort()]
    return wavevector

def wavevector2d(Numofq = 500):
    """ Define Wave Vector for Structure Factors at two dimension """

    wavenumber = np.square(np.arange(Numofq))
    wavevector = []
    for a in range(Numofq):
        for b in range(Numofq):
            d = a**2 + b**2
            if d in wavenumber: 
                wavevector.append(np.array([d, a, b]))
    wavevector = np.ravel(np.array(wavevector))[3:].reshape((-1, 3))
    wavevector = wavevector[wavevector[:, 0].argsort()]   
    return wavevector

def choosewavevector(Numofq, ndim):
    """ Choose Wavevector in dynamic structure factor """

    if ndim == 3:
        return wavevector3d(Numofq)
    elif ndim == 2:
        return wavevector2d(Numofq)

class sq:
    """ Compute static (Partial) structure factors """

    def __init__(self, inputfile, ndim, filetype = 'lammps', moltypes = '', qrange = 10, *arg):
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
        self.qrange = qrange

        self.ParticleNumber     = d.ParticleNumber[0] 
        if d.ParticleNumber[0] != d.ParticleNumber[-1]:
            print ('Warning: ************* Paticle Number Changes **************')
        self.ParticleType   = d.ParticleType
        self.Positions      = d.Positions
        self.SnapshotNumber = d.SnapshotNumber
        self.Boxlength      = d.Boxlength[0]
        if not (d.Boxlength[0] == d.Boxlength[-1]).all():
            print ('Warning: *********Box Length Changed from Dump***********')
        if len(np.unique(self.Boxlength)) != 1:
            print ('Warning: *********Box is not Cubic***********************')
        self.twopidl    = 2 * pi / self.Boxlength[0]  
        self.Boxbounds  = d.Boxbounds[0]
        self.typecounts = np.unique(self.ParticleType[0], return_counts = True) 
        self.Type       = self.typecounts[0]
        self.TypeNumber = self.typecounts[1]
        print ('Particle Type:', self.Type)
        print ('Particle TypeNumber:', self.TypeNumber)
        if np.sum(self.TypeNumber) != self.ParticleNumber:
            print ('Warning: ****** Sum of Indivdual Types is Not the Total Amount*******')

    def getresults(self, outputfile = ''):

        if len(self.Type) == 1:
            return self.Unary(outputfile)

        if len(self.Type) == 2: 
            return self.Binary(outputfile)

        if len(self.Type) == 3: 
            return self.Ternary(outputfile)

        if len(self.Type) == 4: 
            return self.Quarternary(outputfile)

        if len(self.Type) == 5: 
            return self.Quinary(outputfile)

        if len(self.Type) == 6: 
            return self.Senary(outputfile)

        if len(self.Type) > 6:
            return self.Unary(outputfile)

    def Unary(self, outputfile = ''):
        """ Structure Factor of Unary System """
        print ('--------- This is a Unary System ------')

        Numofq = int(self.qrange / self.twopidl)

        wavevector = choosewavevector(Numofq, self.ndim)
        qvalue, qcount = np.unique(wavevector[:, 0], return_counts = True)
        sqresults = np.zeros((len(wavevector[:, 0]), 2)) #the first row accouants for wavenumber

        for n in range(self.SnapshotNumber):
            sqtotal = np.zeros((len(wavevector[:, 0]), 2))
            for i in range(self.ParticleNumber):
                medium   = self.twopidl * (self.Positions[n][i] * wavevector[:, 1:]).sum(axis = 1)
                sqtotal += np.column_stack((np.sin(medium), np.cos(medium)))
            
            sqresults[:, 1] += np.square(sqtotal).sum(axis = 1) / self.ParticleNumber

        sqresults[:, 0]  = wavevector[:, 0]
        sqresults[:, 1]  = sqresults[:, 1] / self.SnapshotNumber

        sqresults = pd.DataFrame(sqresults)
        results   = np.array(sqresults.groupby(sqresults[0]).mean())
        qvalue    = self.twopidl * np.sqrt(qvalue)
        results   = np.column_stack((qvalue, results))
        names = 'q  S(q)'
        if outputfile:
            np.savetxt(outputfile, results, fmt='%.6f', header = names, comments = '')

        print ('--------- Compute S(q) over ------')
        return results, names

    def Binary(self, outputfile = ''):
        """ Structure Factor of Binary System """
        print ('--------- This is a Binary System ------')

        Numofq = int(self.qrange / self.twopidl)

        wavevector = choosewavevector(Numofq, self.ndim)
        qvalue, qcount = np.unique(wavevector[:, 0], return_counts = True)
        sqresults = np.zeros((len(wavevector[:, 0]), 4)) #the first row accouants for wavenumber

        for n in range(self.SnapshotNumber):
            sqtotal = np.zeros((len(wavevector[:, 0]), 2))
            sq11    = np.zeros((len(wavevector[:, 0]), 2))
            sq22    = np.zeros((len(wavevector[:, 0]), 2))
            for i in range(self.ParticleNumber):
                medium   = self.twopidl * (self.Positions[n][i] * wavevector[:, 1:]).sum(axis = 1)
                sqtotal += np.column_stack((np.sin(medium), np.cos(medium)))
                if self.ParticleType[n][i] == 1: sq11 += np.column_stack((np.sin(medium), np.cos(medium)))
                if self.ParticleType[n][i] == 2: sq22 += np.column_stack((np.sin(medium), np.cos(medium)))
            
            sqresults[:, 1] += np.square(sqtotal).sum(axis = 1) / self.ParticleNumber
            sqresults[:, 2] += np.square(sq11).sum(axis = 1) / self.TypeNumber[0]
            sqresults[:, 3] += np.square(sq22).sum(axis = 1) / self.TypeNumber[1]

        sqresults[:, 0]  = wavevector[:, 0]
        sqresults[:, 1:] = sqresults[:, 1:] / self.SnapshotNumber

        sqresults = pd.DataFrame(sqresults)
        results   = np.array(sqresults.groupby(sqresults[0]).mean())
        qvalue    = self.twopidl * np.sqrt(qvalue)
        results   = np.column_stack((qvalue, results))
        names = 'q  S(q)  S11(q)  S22(q)'
        if outputfile:
            np.savetxt(outputfile, results, fmt='%.6f', header = names, comments = '')

        print ('--------- Compute S(q) over ------')
        return results, names

    def Ternary(self, outputfile = ''):
        """ Structure Factor of Ternary System """
        print ('--------- This is a Ternary System ------')  

        Numofq = int(self.qrange / self.twopidl)

        wavevector = choosewavevector(Numofq, self.ndim)
        qvalue, qcount = np.unique(wavevector[:, 0], return_counts = True)
        sqresults = np.zeros((len(wavevector[:, 0]), 5)) #the first row accouants for wavenumber

        for n in range(self.SnapshotNumber):
            sqtotal = np.zeros((len(wavevector[:, 0]), 2))
            sq11    = np.zeros((len(wavevector[:, 0]), 2))
            sq22    = np.zeros((len(wavevector[:, 0]), 2))
            sq33    = np.zeros((len(wavevector[:, 0]), 2))
            for i in range(self.ParticleNumber):
                medium   = self.twopidl * (self.Positions[n][i] * wavevector[:, 1:]).sum(axis = 1)
                sqtotal += np.column_stack((np.sin(medium), np.cos(medium)))
                if self.ParticleType[n][i] == 1: sq11 += np.column_stack((np.sin(medium), np.cos(medium)))
                if self.ParticleType[n][i] == 2: sq22 += np.column_stack((np.sin(medium), np.cos(medium)))
                if self.ParticleType[n][i] == 3: sq33 += np.column_stack((np.sin(medium), np.cos(medium)))
            
            sqresults[:, 1] += np.square(sqtotal).sum(axis = 1) / self.ParticleNumber
            sqresults[:, 2] += np.square(sq11).sum(axis = 1) / self.TypeNumber[0]
            sqresults[:, 3] += np.square(sq22).sum(axis = 1) / self.TypeNumber[1]
            sqresults[:, 4] += np.square(sq33).sum(axis = 1) / self.TypeNumber[2]

        sqresults[:, 0]  = wavevector[:, 0]
        sqresults[:, 1:] = sqresults[:, 1:] / self.SnapshotNumber

        sqresults = pd.DataFrame(sqresults)
        results   = np.array(sqresults.groupby(sqresults[0]).mean())
        qvalue    = self.twopidl * np.sqrt(qvalue)
        results   = np.column_stack((qvalue, results))
        names = 'q  S(q)  S11(q)  S22(q)  S33(q)'
        if outputfile:
            np.savetxt(outputfile, results, fmt='%.6f', header = names, comments = '')

        print ('--------- Compute S(q) over ------')
        return results, names

    def Quarternary(self, outputfile = ''):
        """ Structure Factor of Quarternary System """
        print ('--------- This is a Quarternary System ------')     

        Numofq = int(self.qrange / self.twopidl)

        wavevector = choosewavevector(Numofq, self.ndim)
        qvalue, qcount = np.unique(wavevector[:, 0], return_counts = True)
        sqresults = np.zeros((len(wavevector[:, 0]), 6)) #the first row accouants for wavenumber

        for n in range(self.SnapshotNumber):
            sqtotal = np.zeros((len(wavevector[:, 0]), 2))
            sq11    = np.zeros((len(wavevector[:, 0]), 2))
            sq22    = np.zeros((len(wavevector[:, 0]), 2))
            sq33    = np.zeros((len(wavevector[:, 0]), 2))
            sq44    = np.zeros((len(wavevector[:, 0]), 2))
            for i in range(self.ParticleNumber):
                medium   = self.twopidl * (self.Positions[n][i] * wavevector[:, 1:]).sum(axis = 1)
                sqtotal += np.column_stack((np.sin(medium), np.cos(medium)))
                if self.ParticleType[n][i] == 1: sq11 += np.column_stack((np.sin(medium), np.cos(medium)))
                if self.ParticleType[n][i] == 2: sq22 += np.column_stack((np.sin(medium), np.cos(medium)))
                if self.ParticleType[n][i] == 3: sq33 += np.column_stack((np.sin(medium), np.cos(medium)))
                if self.ParticleType[n][i] == 4: sq44 += np.column_stack((np.sin(medium), np.cos(medium)))
            
            sqresults[:, 1] += np.square(sqtotal).sum(axis = 1) / self.ParticleNumber
            sqresults[:, 2] += np.square(sq11).sum(axis = 1) / self.TypeNumber[0]
            sqresults[:, 3] += np.square(sq22).sum(axis = 1) / self.TypeNumber[1]
            sqresults[:, 4] += np.square(sq33).sum(axis = 1) / self.TypeNumber[2]
            sqresults[:, 5] += np.square(sq44).sum(axis = 1) / self.TypeNumber[3]

        sqresults[:, 0]  = wavevector[:, 0]
        sqresults[:, 1:] = sqresults[:, 1:] / self.SnapshotNumber

        sqresults = pd.DataFrame(sqresults)
        results   = np.array(sqresults.groupby(sqresults[0]).mean())
        qvalue    = self.twopidl * np.sqrt(qvalue)
        results   = np.column_stack((qvalue, results))
        names = 'q  S(q)  S11(q)  S22(q)  S33(q)  S44(q)'
        if outputfile:
            np.savetxt(outputfile, results, fmt='%.6f', header = names, comments = '')
        
        print ('--------- Compute S(q) over ------')
        return results, names

    def Quinary(self, outputfile = ''):
        """ Structure Factor of Qinary System """
        print ('--------- This is a Quinary System ------')       

        Numofq = int(self.qrange / self.twopidl)

        wavevector = choosewavevector(Numofq, self.ndim)
        qvalue, qcount = np.unique(wavevector[:, 0], return_counts = True)
        sqresults = np.zeros((len(wavevector[:, 0]), 7)) #the first row accouants for wavenumber

        for n in range(self.SnapshotNumber):
            sqtotal = np.zeros((len(wavevector[:, 0]), 2))
            sq11    = np.zeros((len(wavevector[:, 0]), 2))
            sq22    = np.zeros((len(wavevector[:, 0]), 2))
            sq33    = np.zeros((len(wavevector[:, 0]), 2))
            sq44    = np.zeros((len(wavevector[:, 0]), 2))
            sq55    = np.zeros((len(wavevector[:, 0]), 2))
            for i in range(self.ParticleNumber):
                medium   = self.twopidl * (self.Positions[n][i] * wavevector[:, 1:]).sum(axis = 1)
                sqtotal += np.column_stack((np.sin(medium), np.cos(medium)))
                if self.ParticleType[n][i] == 1: sq11 += np.column_stack((np.sin(medium), np.cos(medium)))
                if self.ParticleType[n][i] == 2: sq22 += np.column_stack((np.sin(medium), np.cos(medium)))
                if self.ParticleType[n][i] == 3: sq33 += np.column_stack((np.sin(medium), np.cos(medium)))
                if self.ParticleType[n][i] == 4: sq44 += np.column_stack((np.sin(medium), np.cos(medium)))
                if self.ParticleType[n][i] == 5: sq55 += np.column_stack((np.sin(medium), np.cos(medium)))
            
            sqresults[:, 1] += np.square(sqtotal).sum(axis = 1) / self.ParticleNumber
            sqresults[:, 2] += np.square(sq11).sum(axis = 1) / self.TypeNumber[0]
            sqresults[:, 3] += np.square(sq22).sum(axis = 1) / self.TypeNumber[1]
            sqresults[:, 4] += np.square(sq33).sum(axis = 1) / self.TypeNumber[2]
            sqresults[:, 5] += np.square(sq44).sum(axis = 1) / self.TypeNumber[3]
            sqresults[:, 6] += np.square(sq55).sum(axis = 1) / self.TypeNumber[4]

        sqresults[:, 0]  = wavevector[:, 0]
        sqresults[:, 1:] = sqresults[:, 1:] / self.SnapshotNumber

        sqresults = pd.DataFrame(sqresults)
        results   = np.array(sqresults.groupby(sqresults[0]).mean())
        qvalue    = self.twopidl * np.sqrt(qvalue)
        results   = np.column_stack((qvalue, results))
        names = 'q  S(q)  S11(q)  S22(q)  S33(q)  S44(q)  S55(q)'
        if outputfile:
            np.savetxt(outputfile, results, fmt='%.6f', header = names, comments = '')

        print ('--------- Compute S(q) over ------')
        return results, names

    def Senary(self, outputfile = ''):
        """ Structure Factor of Senary System """
        print ('--------- This is a Senary System ------')

        Numofq = int(self.qrange / self.twopidl)

        wavevector = choosewavevector(Numofq, self.ndim)
        qvalue, qcount = np.unique(wavevector[:, 0], return_counts = True)
        sqresults = np.zeros((len(wavevector[:, 0]), 8)) #the first row accouants for wavenumber

        for n in range(self.SnapshotNumber):
            sqtotal = np.zeros((len(wavevector[:, 0]), 2))
            sq11    = np.zeros((len(wavevector[:, 0]), 2))
            sq22    = np.zeros((len(wavevector[:, 0]), 2))
            sq33    = np.zeros((len(wavevector[:, 0]), 2))
            sq44    = np.zeros((len(wavevector[:, 0]), 2))
            sq55    = np.zeros((len(wavevector[:, 0]), 2))
            sq66    = np.zeros((len(wavevector[:, 0]), 2))
            for i in range(self.ParticleNumber):
                medium   = self.twopidl * (self.Positions[n][i] * wavevector[:, 1:]).sum(axis = 1)
                sqtotal += np.column_stack((np.sin(medium), np.cos(medium)))
                if self.ParticleType[n][i] == 1: sq11 += np.column_stack((np.sin(medium), np.cos(medium)))
                if self.ParticleType[n][i] == 2: sq22 += np.column_stack((np.sin(medium), np.cos(medium)))
                if self.ParticleType[n][i] == 3: sq33 += np.column_stack((np.sin(medium), np.cos(medium)))
                if self.ParticleType[n][i] == 4: sq44 += np.column_stack((np.sin(medium), np.cos(medium)))
                if self.ParticleType[n][i] == 5: sq55 += np.column_stack((np.sin(medium), np.cos(medium)))
                if self.ParticleType[n][i] == 6: sq66 += np.column_stack((np.sin(medium), np.cos(medium)))
            
            sqresults[:, 1] += np.square(sqtotal).sum(axis = 1) / self.ParticleNumber
            sqresults[:, 2] += np.square(sq11).sum(axis = 1) / self.TypeNumber[0]
            sqresults[:, 3] += np.square(sq22).sum(axis = 1) / self.TypeNumber[1]
            sqresults[:, 4] += np.square(sq33).sum(axis = 1) / self.TypeNumber[2]
            sqresults[:, 5] += np.square(sq44).sum(axis = 1) / self.TypeNumber[3]
            sqresults[:, 6] += np.square(sq55).sum(axis = 1) / self.TypeNumber[4]
            sqresults[:, 7] += np.square(sq66).sum(axis = 1) / self.TypeNumber[5]

        sqresults[:, 0]  = wavevector[:, 0]
        sqresults[:, 1:] = sqresults[:, 1:] / self.SnapshotNumber

        sqresults = pd.DataFrame(sqresults)
        results   = np.array(sqresults.groupby(sqresults[0]).mean())
        qvalue    = self.twopidl * np.sqrt(qvalue)
        results   = np.column_stack((qvalue, results))
        names = 'q  S(q)  S11(q)  S22(q)  S33(q)  S44(q)  S55(q)  S66(q)'
        if outputfile:
            np.savetxt(outputfile, results, fmt='%.6f', header = names, comments = '')

        print ('--------- Compute S(q) over ------')
        return results, names
