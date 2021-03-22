#!/usr/bin/python
# coding = utf-8
#This module is part of an analysis package

Authorinfo = """
             ------------------Name: Yuan-Chao Hu--------------
             --------------Email: ychu0213@gmail.com-----------
             ----------Web: https://yuanchaohu.github.io/------
             """

Docstr = """
             Parse the dipole vector from LAMMPS dump file with
             'id type x y mux muy'
         """

import numpy as np
import subprocess


def GetDipoleVector(inputfile, ndim=2, outputfile='', SnapshotNumber=0):
    """Get the Dipole Vector from LAMMPS dump file with [id type x y mux muy]"""

    fin  = open(inputfile, 'r')
    fout = open(outputfile, 'w')

    if not SnapshotNumber:
        cmdline = 'grep -o "ITEM: TIMESTEP" %s | wc -l'%inputfile
        process = subprocess.Popen(cmdline, stdout=subprocess.PIPE, stderr=None, shell=True)
        SnapshotNumber = int(process.communicate()[0].decode())
    print ('Snapshot Number: %s'%SnapshotNumber)

    for n in range(SnapshotNumber):
        fin.readline()
        item = fin.readline()
        fout.write('Step: %s'%item)
        fin.readline()
        Natom = int(fin.readline())
        fout.write('Natom: %d\n'%Natom)
        if ndim == 2:
            fout.write('id type mux muy\n')
        elif ndim == 3:
            fout.write('id type mux muy muz\n')

        for i in range(5):
            fin.readline()
        
        for i in range(Natom):
            item = fin.readline().split()
            fout.write('%s %s '%(item[0], item[1]))
            fout.write(' '.join(item[-ndim:]))
            fout.write('\n')

    fin.close()
    fout.close()
    print ('-------writing dipole vectors done------')
