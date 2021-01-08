#!/usr/bin/python
# coding = utf-8
#This module is part of an analysis package

Authorinfo = """
             ------------------Name: Yuan-Chao Hu--------------
             --------------Email: ychu0213@gmail.com-----------
             ----------The University of Tokyo (2020)----------
             """

Docstr = """
         This module calculates the autocorrelation functions of
         various quantities, such as density, current, velocity...
         
         BOTH positions and velocities may be needed as input
         large samples are usually needed to access small wavenumber

         ref. Shintani and Tanaka, Nat. Mater. (2008) [current-current autocorrelation]
         ref. Cheng and Frenkel, Phys. Rev. Lett. 125, 130602 (2020) [density autocorrelation]
         """

import pandas as pd 
import numpy as np
from math import pi, cos, sin 

def skip_file(f):
    """skip snapshot in a dump file"""

    f.readline()
    f.readline()
    f.readline()
    NumofAtom = int(f.readline())
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    f.readline()
    for i in range(NumofAtom):
        f.readline()

def read_file(f, ndim = 2):
    """read particle positions and velocities of single configuration
       The box lengths in different directions should be the same
       use 'dump_modify sort id' for output
    
    input:  id type x y (z) vx vy (vz)
    return: timestep
            box size 
            position
            velocity
    """

    BoxLength = np.zeros(ndim)
    #reading header of lammps dump file
    f.readline()
    TimeStep = int(f.readline())
    f.readline()
    NumofAtom = int(f.readline())
    f.readline()
    item = f.readline().split()
    BoxLength[0] = float(item[1]) - float(item[0])
    item = f.readline().split()
    BoxLength[1] = float(item[1]) - float(item[0])
    if ndim == 3:
        item = f.readline().split()
        BoxLength[2] = float(item[1]) - float(item[0])
    else:
        f.readline()
    f.readline()

    #reading particle information
    Positions  = np.zeros((NumofAtom, ndim))
    Velocities = np.zeros((NumofAtom, ndim))
    for i in range(NumofAtom):
        item = f.readline().split()
        item = [float(j) for j in item]
        if len(item) == (2+ndim*2):
            #id type x y (z) vx vy (vz)
            Positions[int(item[0])-1]  = item[2:ndim+2]
            Velocities[int(item[0])-1] = item[-ndim:]
        elif len(item) == ndim*2:
            #x y (z) vx vy (vz)
            Positions[i] = item[:ndim]
            Velocities[i] = item[-ndim:]
        elif len(item) == ndim:
            #x y (z) or vx vy (vz)
            Positions[i] = item[:]
            Velocities[i] = item[:]
    
    return TimeStep, NumofAtom, BoxLength, Positions, Velocities


def velocity_autocorrelation(inputfile, numofconfig, ndim=2, Nevery=1, dt=0.002, mass=1.0, outputfile=''):
    """calculate velocity-velocity autocorrelation function
    
    if the mass of different particle types is not the same,
    it should be a column numpy array in the shape of (numofatom, 1)
    Nevery defines the frequency of reading a trajectory file
    """

    #read dump file
    f = open(inputfile)
    print('reading input file...')
    AllTime  = []
    velocity = []
    for n in range(numofconfig):
        if n%Nevery == 0:
            t, NumofAtom, _, _, Velocities = read_file(f, ndim)
            AllTime.append(t)
            velocity.append(Velocities)
        else:
            skip_file(f)
    f.close()

    numofconfig = numofconfig // Nevery
    AllTime = np.array(AllTime)[:numofconfig]
    results  = np.zeros(numofconfig)
    counts   = np.zeros(numofconfig)

    for n in range(numofconfig):
        print (n)
        for nn in range(n): #time interval
            results[nn] += (mass * velocity[n] * velocity[n-nn]).sum()
            counts[nn]  += 1
    
    condition = counts > 100#(numofconfig//2)
    results = results[condition] / counts[condition] / NumofAtom
    results = results / results[0] #normalization
    Times   = (AllTime[condition] - AllTime[0]) * dt  
    np.savetxt(outputfile, np.column_stack((Times, results)), fmt='%.3f %.6f', comments='', header='t C_V(t)')
    
    print ('-----Calculate velocity autocorrelation function DONE-----')


def velocity_angular_autocorrelation(inputfile, numofconfig, ndim=2, Nevery=1, dt=0.002, mass=1.0, inertia=0, outputfile=''):
    """calculate velocity-velocity including angular motion autocorrelation function
    
    if the mass of different particle types is not the same,
    it should be a column numpy array in the shape of (numofatom, 1)
    Nevery defines the frequency of reading a trajectory file

    inertia is the moment of inertia of spheres
    see http://hyperphysics.phy-astr.gsu.edu/hbase/mi.html
    The definition of inertia should be the same as mass
    """

    #read dump file
    f = open(inputfile)
    print('reading input file...')
    AllTime = []
    velocity = []
    angular  = [] 
    for n in range(numofconfig):
        if n % Nevery == 0:
            TimeStep,NumofAtom,BoxLength,Positions,Velocities=read_file(f, ndim)
            AllTime.append(TimeStep)
            velocity.append(Velocities)
            angular.append(Positions)
        else:
            skip_file(f)
    f.close()

    numofconfig = numofconfig // Nevery
    AllTime = np.array(AllTime)[:numofconfig]
    results = np.zeros(numofconfig)
    counts = np.zeros(numofconfig)

    for n in range(numofconfig):
        print(n)
        for nn in range(n+1):  # time interval
            results[nn] += (mass * velocity[n] * velocity[n-nn]).sum() #translational
            results[nn] += (inertia*angular[n]*angular[n-nn]).sum() #rotational
            counts[nn]  += 1

    condition = counts > 100  # (numofconfig//2)
    results = results[condition] / counts[condition] / NumofAtom
    results = results / results[0]  # normalization
    Times = (AllTime[condition] - AllTime[0]) * dt
    np.savetxt(outputfile, np.column_stack((Times, results)),
               fmt='%.3f %.6f', comments='', header='t C_V(t)')

    print('-----Calculate velocity-angular autocorrelation function DONE-----')

def Current_autocorrelation(inputfile, numofconfig, qvector, Nevery=1, dt=0.002, Tfile='', Lfile=''):
    """Calculate longitudinal and transverse current-current autocorrelation function
    
    one wave vector is given for a computation [qvector], like [1, 0]
    Nevery defines the frequency of reading a trajectory file
    """

    qvector = np.array(qvector)
    ndim = len(qvector)

    #read dump file
    f = open(inputfile)
    print ('reading input file...')
    AllTime   = []
    Pos_Atoms = []
    Vel_Atoms = []
    for n in range(numofconfig):
        if n%Nevery == 0:
            t, NumofAtom, BoxLength, Positions, Velocities = read_file(f, ndim)
            AllTime.append(t)
            Pos_Atoms.append(Positions)
            Vel_Atoms.append(Velocities)
        else:
            skip_file(f)
    f.close()
    
    #define effective variables
    numofconfig = numofconfig // Nevery
    twopidL = 2.0 * pi / BoxLength
    qvector = twopidL * qvector
    qvalue  = np.linalg.norm(qvector)
    unitqvector = qvector / qvalue
    AllTime = np.array(AllTime)[:numofconfig]

    #longitudinal L
    vxsin_L, vxcos_L = np.zeros(numofconfig), np.zeros(numofconfig)
    vysin_L, vycos_L = np.zeros(numofconfig), np.zeros(numofconfig)
    if ndim==3:
        vzsin_L, vzcos_L = np.zeros(numofconfig), np.zeros(numofconfig)
    #transverse T
    vxsin_T, vxcos_T = np.zeros(numofconfig), np.zeros(numofconfig)
    vysin_T, vycos_T = np.zeros(numofconfig), np.zeros(numofconfig)
    if ndim==3:
        vzsin_T, vzcos_T = np.zeros(numofconfig), np.zeros(numofconfig)
    #store results
    results_L = np.zeros(numofconfig)
    results_T = np.zeros(numofconfig)
    counts    = np.zeros(numofconfig)
    
    for n in range(numofconfig):
        print (n)       
        for i in range(NumofAtom):
            term0 = np.dot(Pos_Atoms[n][i], qvector) #a
            value_sin = sin(term0)
            value_cos = cos(term0)

            #longitudinal L
            term1 = np.dot(Vel_Atoms[n][i], unitqvector) * unitqvector
            vxsin_L[n] += term1[0] * value_sin
            vxcos_L[n] += term1[0] * value_cos
            vysin_L[n] += term1[1] * value_sin
            vycos_L[n] += term1[1] * value_cos
            if ndim ==3:
                vzsin_L[n] += term1[2] * value_sin
                vzcos_L[n] += term1[2] * value_cos

            #transverse T
            term2 = Vel_Atoms[n][i] - term1
            vxsin_T[n] += term2[0] * value_sin
            vxcos_T[n] += term2[0] * value_cos 
            vysin_T[n] += term2[1] * value_sin
            vycos_T[n] += term2[1] * value_cos
            if ndim == 3:
                vzsin_T[n] += term2[2] * value_sin
                vzcos_T[n] += term2[2] * value_cos
        
        for nn in range(n): #time interval
            results_L[nn] += vxsin_L[n]*vxsin_L[n-nn]+vxcos_L[n]*vxcos_L[n-nn]
            results_L[nn] += vysin_L[n]*vysin_L[n-nn]+vycos_L[n]*vycos_L[n-nn]
            if ndim ==3:
                results_L[nn] += vzsin_L[n]*vzsin_L[n-nn]+vzcos_L[n]*vzcos_L[n-nn]
            
            results_T[nn] += vxsin_T[n]*vxsin_T[n-nn]+vxcos_T[n]*vxcos_T[n-nn]
            results_T[nn] += vysin_T[n]*vysin_T[n-nn]+vycos_T[n]*vycos_T[n-nn]
            if ndim == 3:
                results_T[nn] += vzsin_T[n]*vzsin_T[n-nn]+vzcos_T[n]*vzcos_T[n-nn]
            
            counts[nn] += 1
        
    condition = counts > 20#(numofconfig//2)
    results_L = results_L[condition] / counts[condition] / NumofAtom
    results_L = results_L / results_L[0]
    results_T = results_T[condition] / counts[condition] / NumofAtom
    results_T = results_T / results_T[0]
    Times = (AllTime[condition] - AllTime[0]) * dt

    if Lfile:
        np.savetxt(Lfile, np.column_stack((Times, results_L)), fmt='%0.3f %.6f', comments='q=%.6f\n'%qvalue, header='t C_L(q, t)')
    
    if Tfile:
        np.savetxt(Tfile, np.column_stack((Times, results_T)), fmt='%0.3f %.6f', comments='q=%.6f\n'%qvalue, header='t C_T(q, t)')

    print ('-----calculate current-current autocorrelation function DONE-----')
    return qvalue, Times, results_L, results_T


def Density_autocorrelation(inputfile, numofconfig, qvector, Nevery=1, dt=0.002, outputfile=''):
    """calculate time dependent density-density correlation function
    
    one wave vector is given for a computation [qvector], like [1, 0]
    Nevery defines the frequency of reading a trajectory file
    Ref. Daan Frenkel Phys. Rev. Lett. (2020)
    """

    qvector = np.array(qvector)
    ndim = len(qvector)
    
    #read dump file
    f = open(inputfile)
    print ('reading input file...')
    AllTime = []
    Pos_Atoms = []
    for n in range(numofconfig):
        if n%Nevery == 0:
            t, NumofAtom, BoxLength, Positions, _ = read_file(f, ndim)
            AllTime.append(t)
            Pos_Atoms.append(Positions)
        else:
            skip_file(f)
    f.close()

    #define effective variables
    numofconfig = numofconfig // Nevery
    twopidL = 2.0 * pi / BoxLength
    qvector = twopidL * qvector
    qvalue  = np.linalg.norm(qvector)
    AllTime = np.array(AllTime)[:numofconfig]
    Volume  = np.prod(BoxLength)

    sin_part = np.zeros(numofconfig)
    cos_part = np.zeros(numofconfig)
    results  = np.zeros(numofconfig)
    counts   = np.zeros(numofconfig)

    for n in range(numofconfig):
        print (n)
        for i in range(NumofAtom):
            theta = np.dot(qvector, Pos_Atoms[n][i])
            sin_part[n] += sin(theta)
            cos_part[n] += cos(theta)
        
        for nn in range(n): #time interval
            results[nn] += -sin_part[n]*sin_part[n-nn]+cos_part[n]*cos_part[n-nn]
            counts[nn]  += 1
        
    condition = counts > 100
    results = results[condition] / counts[condition] / Volume / Volume / NumofAtom
    results = results / results[0]
    Times = (AllTime[condition] - AllTime[0]) * dt 

    #t integral
    from scipy import integrate
    integrals = integrate.cumtrapz(results, Times, initial=1.0)

    if outputfile:
        np.savetxt(outputfile, np.column_stack((Times, results, integrals)),
                   fmt='%.3f %.6f %.6f', comments='q=%.6f\n' % qvalue, header='t Corr Integral')


#-----------------------------------------------------------------------------------
#-------------separate file reading process and the calculation---------------------
#-----------------------------------------------------------------------------------

def Get_Input(inputfile, numofconfig, Nevery=1, dt=0.002, ndim=2):
    """get necessary information from a large dump for calculations over multiple wavevectors"""

    #read dump file
    f = open(inputfile)
    print('reading input file...')
    AllTime = []
    Pos_Atoms = []
    Vel_Atoms = []
    for n in range(numofconfig):
        if n % Nevery == 0:
            t, NumofAtom, BoxLength, Positions, Velocities = read_file(f, ndim)
            AllTime.append(t)
            Pos_Atoms.append(Positions)
            Vel_Atoms.append(Velocities)
        else:
            skip_file(f)
    f.close()

    numofconfig = len(Pos_Atoms)
    AllTime = np.array(AllTime)[:numofconfig]
    AllTime = (AllTime - AllTime[0]) * dt

    print('reading input file DONE')

    return AllTime, Pos_Atoms, Vel_Atoms, BoxLength

def Current_autocorrelation_Noreading(AllTime, Pos_Atoms, Vel_Atoms, BoxLength, qvector, Tfile='', Lfile=''):
    """Calculate longitudinal and transverse current-current autocorrelation function
    
    one wave vector is given for a computation [qvector], like [1, 0]
    Nevery defines the frequency of reading a trajectory file
    """

    qvector = np.array(qvector)
    ndim = len(qvector)

    #define effective variables
    numofconfig = len(AllTime)
    NumofAtom = Pos_Atoms[0].shape[0]
    twopidL = 2.0 * pi / BoxLength
    qvector = twopidL * qvector
    qvalue = np.linalg.norm(qvector)
    unitqvector = qvector / qvalue

    #longitudinal L
    vxsin_L, vxcos_L = np.zeros(numofconfig), np.zeros(numofconfig)
    vysin_L, vycos_L = np.zeros(numofconfig), np.zeros(numofconfig)
    if ndim==3:
        vzsin_L, vzcos_L = np.zeros(numofconfig), np.zeros(numofconfig)
    #transverse T
    vxsin_T, vxcos_T = np.zeros(numofconfig), np.zeros(numofconfig)
    vysin_T, vycos_T = np.zeros(numofconfig), np.zeros(numofconfig)
    if ndim==3:
        vzsin_T, vzcos_T = np.zeros(numofconfig), np.zeros(numofconfig)
    #store results
    results_L = np.zeros(numofconfig)
    results_T = np.zeros(numofconfig)
    counts = np.zeros(numofconfig)

    for n in range(numofconfig):
        print(n)
        for i in range(NumofAtom):
            term0 = np.dot(Pos_Atoms[n][i], qvector)  # a
            value_sin = sin(term0)
            value_cos = cos(term0)

            #longitudinal L
            term1 = np.dot(Vel_Atoms[n][i], unitqvector) * unitqvector
            vxsin_L[n] += term1[0] * value_sin
            vxcos_L[n] += term1[0] * value_cos
            vysin_L[n] += term1[1] * value_sin
            vycos_L[n] += term1[1] * value_cos
            if ndim == 3:
                vzsin_L[n] += term1[2] * value_sin
                vzcos_L[n] += term1[2] * value_cos

            #transverse T
            term2 = Vel_Atoms[n][i] - term1
            vxsin_T[n] += term2[0] * value_sin
            vxcos_T[n] += term2[0] * value_cos
            vysin_T[n] += term2[1] * value_sin
            vycos_T[n] += term2[1] * value_cos
            if ndim == 3:
                vzsin_T[n] += term2[2] * value_sin
                vzcos_T[n] += term2[2] * value_cos

        for nn in range(n):  # time interval
            results_L[nn] += vxsin_L[n]*vxsin_L[n-nn]+vxcos_L[n]*vxcos_L[n-nn]
            results_L[nn] += vysin_L[n]*vysin_L[n-nn]+vycos_L[n]*vycos_L[n-nn]
            if ndim == 3:
                results_L[nn] += vzsin_L[n] * \
                    vzsin_L[n-nn]+vzcos_L[n]*vzcos_L[n-nn]

            results_T[nn] += vxsin_T[n]*vxsin_T[n-nn]+vxcos_T[n]*vxcos_T[n-nn]
            results_T[nn] += vysin_T[n]*vysin_T[n-nn]+vycos_T[n]*vycos_T[n-nn]
            if ndim == 3:
                results_T[nn] += vzsin_T[n] * \
                    vzsin_T[n-nn]+vzcos_T[n]*vzcos_T[n-nn]

            counts[nn] += 1

    condition = counts > 20  # (numofconfig//2)
    results_L = results_L[condition] / counts[condition] / NumofAtom
    results_L = results_L / results_L[0]
    results_T = results_T[condition] / counts[condition] / NumofAtom
    results_T = results_T / results_T[0]

    if Lfile:
        np.savetxt(Lfile, np.column_stack((AllTime[condition], results_L)), fmt='%0.3f %.6f',
                   comments='q=%.6f\n' % qvalue, header='t C_L(q, t)')

    if Tfile:
        np.savetxt(Tfile, np.column_stack((AllTime[condition], results_T)), fmt='%0.3f %.6f',
                   comments='q=%.6f\n' % qvalue, header='t C_T(q, t)')

    print('-----calculate current-current autocorrelation function DONE-----')

    #return qvalue, AllTime, results_L, results_T
