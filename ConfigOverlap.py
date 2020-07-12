#!/usr/bin/python
# coding = utf-8
#This module is part of an analysis package

Authorinfo = """
             ------------------Name: Yuan-Chao Hu--------------
             --------------Email: ychu0213@gmail.com-----------
             ----------Web: https://yuanchaohu.github.io/------
             """

Docstr = """
         This module calculates the overlap between two configurations, 
         i.e. the similarity in configs

         The code accounts for both orthogonal and triclinic boxes by using h-matrix to 
         deal with periodic boundary conditions. This is important in dealing with
         particle distance in triclinic cells
         """

import numpy  as np 
import pandas as pd 
from dump import readdump
import os 
from ParticleNeighbors import Voropp

def ConfigOverlap_Single(inputfile, ndim, binsize, outputfile, filetype = 'lammps', moltypes = ''):
    """
     Calculate the overlap of configurations to get the similarity of configurations
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

    d = readdump(inputfile, ndim, filetype, moltypes)
    d.read_onefile()
    positions      = d.Positions
    particlenumber = d.ParticleNumber[0]
    snapshotnumber = d.SnapshotNumber
    boxlength      = d.Boxlength[0] #configurations box should be the same to be compared

    CellNumber     = int( 1.25 * boxlength.max() / binsize) #to embrace the whole box 
    Cell           = np.zeros((CellNumber, CellNumber, CellNumber, snapshotnumber))
    iatom          = np.zeros((particlenumber, ndim, snapshotnumber), dtype = np.int)
    f              = open(outputfile, 'w')
    f.write('overall     selfpart \n')

    for i in range(snapshotnumber):
        iatom[:, :, i] = np.rint(positions[i] / binsize)
        for j in range(particlenumber):
            Cell[iatom[j, 0, i], iatom[j, 1, i], iatom[j, 2, i], i] += 1
        if (Cell > 1).any():
            print ('Warning: More than one atom in a cell, please reduce binsize')

    for i in range(snapshotnumber - 1):
        for j in range(1, snapshotnumber - i):
            overall   = (Cell[iatom[:, 0, i], iatom[:, 1, i], iatom[:, 2, i], i] * 
                         Cell[iatom[:, 0, i], iatom[:, 1, i], iatom[:, 2, i], i + j]).sum() #/ particlenumber
            selfpart  = (Cell[iatom[:, 0, i], iatom[:, 1, i], iatom[:, 2, i], i] == 
                         Cell[iatom[:, 0, i], iatom[:, 1, i], iatom[:, 2, i], i + j]).sum() #/ particlenumber
            f.write('{:.6f}    {:.6f} \n'.format(overall, selfpart))
    f.close()





#ConfigOverlap_Single('../../dump/ZrCuAl.xs.lammpstrj', 3, 0.68, 'test.dat', results_path = '../')   
    # results = np.column_stack((np.arange(particlenumber) + 1, results))
    # strain  = np.arange(snapshotnumber) * timestep * dt * strainrate
    # results = np.vstack((strain, results))
    # names   = 'id   The_first_row_is_the_strain.0isNAN'
    # fformat = '%d ' + '%.6f ' * (snapshotnumber - 1)
    # np.savetxt(results_path + outputfile, results, fmt = fformat, header = names, comments = '')
    # print ('------ Calculate Von Mises Strain Over -------')
    # return results
