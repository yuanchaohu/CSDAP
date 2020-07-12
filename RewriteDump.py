#coding = utf-8

Authorinfo = """
             ------------------Name: Yuan-Chao Hu--------------
             --------------Email: ychu0213@gmail.com-----------
             ----------Web: https://yuanchaohu.github.io/------
             """

#Prepare the header part of different file types

import numpy as np 

def lammps(step, atomnum, boxbounds, addson = ''):
    """write the 9 headers of lammps dump file"""

    line = 'ITEM: TIMESTEP\n'
    line += str(step) + '\n'

    line += 'ITEM: NUMBER OF ATOMS\n'
    line += str(atomnum) + '\n'

    line += 'ITEM: BOX BOUNDS pp pp pp\n'
    line += '%.6f %.6f\n' %(boxbounds[0, 0], boxbounds[0, 1])
    line += '%.6f %.6f\n' %(boxbounds[1, 0], boxbounds[1, 1])
    if boxbounds.shape[0] == 3:
        line += '%.6f %.6f\n' %(boxbounds[2, 0], boxbounds[2, 1])
    else:
        line += '%.6f %.6f\n' %(-0.5, 0.5)

    line += 'ITEM: ATOMS id type x y z %s\n' %addson

    return line

def lammpsdata(atomnum, atomtypes, boxbounds):
    """write the headers of lammps data file"""

    line = '#LAMMPS data file\n\n'
    line += '%d atoms\n' %(atomnum)
    line += '%d atom types\n\n' %(atomtypes)
    line += '%.6f %.6f xlo xhi\n' %(boxbounds[0, 0], boxbounds[0, 1])
    line += '%.6f %.6f ylo yhi\n' %(boxbounds[1, 0], boxbounds[1, 1])
    line += '%.6f %.6f zlo zhi\n' %(boxbounds[2, 0], boxbounds[2, 1])
    line += '\nAtoms #atomic\n'

    return line