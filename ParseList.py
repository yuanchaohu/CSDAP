#!/usr/bin/python
# coding = utf-8
#This module is part of an analysis package

Authorinfo = """
             ------------------Name: Yuan-Chao Hu--------------
             --------------Email: ychu0213@gmail.com-----------
             ----------Web: https://yuanchaohu.github.io/------
             """

Docstr = """
             Reading partcles' Voronoi analysis results from snapshots of molecular simulations (LAMMPS)

             Rewrite neighbor list and facearea list in a proper format for further analysis
         """

import numpy as np
import re

def readall(fnfile, fffile, fread, ParticleNumber, Snapshotnumber):
    """ Read and rewrite all voronoi results from dump file

        This is suitable for the case when the Voronoi neighbor list and 
        facearea list are calculated during simulation (i.e. LAMMPS)
        by using the codes in lammps script:
        ------------------------------------------------------------------
        compute voro all voronoi/atom neighbors yes
        dump name all local N dumpfile index c_voro[1] c_voro[2] c_voro[3]
        ------------------------------------------------------------------
    """
    fn = open(fnfile, 'w')  #neighbor list file 
    ff = open(fffile, 'w')  #facearea list file 
    f  = open(fread, 'r')   #dump file contains the above information from lammps
    for n in range(Snapshotnumber):
        readone(fn, ff, f, ParticleNumber)

    f.close()
    fn.close()
    ff.close()
    print ('-----Got all neighbor list and facearea list from dump file-----')

def readone(fn, ff, f, ParticleNumber):
    """ Read One Snapshot a Time """ 

    for i in range(3):
        f.readline()
    NumEntries = int(f.readline())
    for i in range(5):
        f.readline()
    
    neighbor = np.zeros((ParticleNumber, 50), dtype = np.int)
    facearea = np.zeros((ParticleNumber, 50))
    for i in range(NumEntries):
        item = f.readline().split()
        neighbor[int(item[1]) - 1, 0] += 1
        neighbor[int(item[1]) - 1, neighbor[int(item[1]) - 1, 0]] = int(item[2])
        facearea[int(item[1]) - 1, 0] += 1       
        facearea[int(item[1]) - 1, neighbor[int(item[1]) - 1, 0]] = float(item[3])

    neighbor = neighbor[:, :neighbor[:, 0].max() + 1]
    neighbor = np.column_stack((np.arange(ParticleNumber) + 1, neighbor))
    facearea = facearea[:, :int(facearea[:, 0].max()) + 1]
    facearea = np.column_stack((np.arange(ParticleNumber) + 1, facearea))

    np.set_printoptions(threshold = np.inf, linewidth = np.inf)
    fn.write('id     cn     neighborlist\n')
    fn.write(re.sub('[\[\]]', ' ', np.array2string(neighbor) + '\n'))
    ff.write('id     cn     facearea-list\n')
    ff.write(re.sub('[\[\]]', ' ', np.array2string(facearea, suppress_small = True, formatter={'float_kind':lambda x: "%.4f" % x}) + '\n'))
