#!/usr/bin/python
# coding = utf-8
#This module is part of an analysis package

Authorinfo = """
             ------------------Name: Yuan-Chao Hu--------------
             --------------Email: ychu0213@gmail.com-----------
             ----------Web: https://yuanchaohu.github.io/------
             """

Docstr = """
             Reading partcles' Neighbor list and Voronoi polyhedron facearea 
             from the output of Voro++ Package analysis

             Voronoi tessellation can be carried out use the provided script 'Voronoi.sh'
             Voropp() is suitable for both data

             ----------------------------------------------------------------------------
             Two new methods to identify the nearest neighbors are also included
             1) identify the N nearest neighbors, such as N = 12 (default)
             2) identify the nearest neighbors by considering a global cutoff distance r_cut
             The atom index in the neighbor list is sorted by the corresponding distance
             These two functions output file format that can be read by using Voropp()

         """

import numpy as np 

def Voropp(f, ParticleNumber, Nmax=100):
    """
    Read Neighbor list data from the results of Voro++ Package
    &&&&&&&&&
    Read facearea list data from the results of Voro++ Package
    or similarly, as long as the format is compatible

    Nmax: the maximum number of neighboring atoms to consider

    Read One Snapshot a time to save computer memory
    If you have multiple snapshots, you can import this function in a loop
    f = open(filename, 'r')
    The Voronoi analysis can be carried out use the provided shell secipt 'voronoi.sh' or Voronoi module
    """

    header  = f.readline().split()  #header
    results = np.zeros((ParticleNumber, Nmax+1))

    for n in range(ParticleNumber):
        item = f.readline().split()
        atom_index = int(item[0]) - 1
        if int(item[1]) <= Nmax:
            results[atom_index, 0] = float(item[1])
            results[atom_index, 1:(int(item[1]) + 1)] = [float(j) - 1 for j in item[2:(int(item[1]) + 2)]]
        else:
            results[atom_index, 0] = Nmax
            results[atom_index, 1:Nmax+1] = [float(j) - 1 for j in item[2:Nmax+2]]
            if n == 0: 
                print ('*******Too Many neighbors [>%d]*********'%Nmax)
                print ('-----warning: not for unsorted neighbor list-----')
        #Be attention to the '-1' after '=', all particle id has been reduced by 1
        #Please be careful when you are reading other data, like face area
        #you should first transform the data back before computation in other codes

    max_CN = int(results[:, 0].max())
    if max_CN < Nmax: 
        results = results[:, :max_CN + 1] #save storage

    if 'neighborlist' in header:  #neighbor list should be integer
        results = results.astype(np.int)
    
    return results


def Nnearests(dumpfile, ndim = 3, filetype = 'lammps', moltypes = '', N = 12, ppp = [1,1,1],  fnfile = 'neighborlist.dat'):
    """Get the N nearest neighbors around a particle
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

    from dump import readdump
    import re

    d = readdump(dumpfile, ndim, filetype, moltypes)
    d.read_onefile()

    fneighbor = open(fnfile, 'w')
    for n in range(d.SnapshotNumber):
        hmatrixinv = np.linalg.inv(d.hmatrix[n])
        Positions  = d.Positions[n]
        neighbor   = np.zeros((d.ParticleNumber[n], 2 + N), dtype = np.int)
        neighbor[:, 0] = np.arange(d.ParticleNumber[n]) + 1
        neighbor[:, 1] = N 

        for i in range(d.ParticleNumber[n]):
            RIJ      = Positions - Positions[i]
            matrixij = np.dot(RIJ, hmatrixinv)
            RIJ      = np.dot(matrixij - np.rint(matrixij) * ppp, d.hmatrix[n]) #remove PBC

            RIJ_norm = np.linalg.norm(RIJ, axis = 1)
            nearests = np.argpartition(RIJ_norm, N + 1)[:N + 1]
            nearests = nearests[RIJ_norm[nearests].argsort()]
            neighbor[i, 2:] = nearests[1:] + 1

    
        np.set_printoptions(threshold = np.inf, linewidth = np.inf)
        fneighbor.write('id     cn     neighborlist\n')
        fneighbor.write(re.sub('[\[\]]', ' ', np.array2string(neighbor) + '\n'))

    fneighbor.close()
    print ('---Calculate %d nearest neighbors done---' %N)

def cutoffneighbors(dumpfile, r_cut, ndim = 3, filetype = 'lammps', moltypes = '', ppp = [1,1,1], fnfile = 'neighborlist.dat'):
    """Get the nearest neighbors around a particle by setting a cutoff distance r_cut
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

    from dump import readdump
    import re

    d = readdump(dumpfile, ndim, filetype, moltypes)
    d.read_onefile()

    fneighbor = open(fnfile, 'w')
    for n in range(d.SnapshotNumber):
        hmatrixinv = np.linalg.inv(d.hmatrix[n])
        Positions  = d.Positions[n]
        neighbor   = np.arange(d.ParticleNumber[n]).astype(np.int32)
        fneighbor.write('id     cn     neighborlist\n')
        for i in range(d.ParticleNumber[n]):
            RIJ      = Positions - Positions[i]
            matrixij = np.dot(RIJ, hmatrixinv)
            RIJ      = np.dot(matrixij - np.rint(matrixij) * ppp, d.hmatrix[n]) #remove PBC

            RIJ_norm = np.linalg.norm(RIJ, axis = 1)
            nearests = neighbor[RIJ_norm <= r_cut]
            CN  = nearests.shape[0] - 1
            nearests = nearests[RIJ_norm[nearests].argsort()]
            nearests = nearests[1:] + 1

            fneighbor.write('%d %d ' %(i + 1, CN))
            fneighbor.write(' '.join(map(str, nearests)))
            fneighbor.write('\n')

    fneighbor.close()
    print ('---Calculate nearest neighbors with r_cut = %.6f done---' %r_cut)

def cutoffneighbors_atomtypes(dumpfile, r_cut, ndim=3, filetype='lammps', moltypes='', ppp=[1,1,1], fnfile='neighborlist.dat'):
    """Get the nearest neighbors around a particle by setting a cutoff distance r_cut
        The keyword filetype is used for different MD engines
        It has four choices:
        'lammps' (default)

        'lammpscenter' (lammps molecular dump with known atom type of each molecule center)
        moltypes is a dict mapping center atomic type to molecular type
        moltypes is also used to select the center atom 
        such as moltypes = {3: 1, 5: 2}

        'gsd' (HOOMD-blue standard output for static properties)
    
        'gsd_dcd' (HOOMD-blue outputs for static and dynamic properties)

        **r_cut: a numpy array for all atom pairs, for example for a binary system should be 
        [1-1, 1-2
         2-1, 2-2]
        **
    """

    from dump import readdump
    import re

    if type(r_cut) is not np.ndarray:
        errorinfo  = "---input r_cut type error: please give a numpy array over all pairs\n"
        errorinfo += "shape of r_cut input is (atom_type_number, atom_type_number)---"
        raise IOError(errorinfo)
    
    d = readdump(dumpfile, ndim, filetype, moltypes)
    d.read_onefile()

    if r_cut.shape[0] != np.unique(d.ParticleType[0]).shape[0]:
        errorinfo = '---Wrong atom_type_number for input r_cut---'
        raise IOError(errorinfo)

    #define cutoffs for each pair based on particle type
    atom_type_number = r_cut.shape[0]
    cutoffs = np.zeros((atom_type_number, d.ParticleNumber[0]))
    for i in range(atom_type_number):
        for j in range(d.ParticleNumber[0]):
            cutoffs[i, j] = r_cut[i, d.ParticleType[0][j]-1]

    fneighbor = open(fnfile, 'w')
    for n in range(d.SnapshotNumber):
        hmatrixinv = np.linalg.inv(d.hmatrix[n])
        Positions  = d.Positions[n]
        neighbor   = np.arange(d.ParticleNumber[n]).astype(np.int32)
        fneighbor.write('id     cn     neighborlist\n')
        for i in range(d.ParticleNumber[n]):
            RIJ      = Positions - Positions[i]
            matrixij = np.dot(RIJ, hmatrixinv)
            RIJ      = np.dot(matrixij - np.rint(matrixij) * ppp, d.hmatrix[n]) #remove PBC
            RIJ_norm = np.linalg.norm(RIJ, axis = 1)
    
            i_cutoffs = cutoffs[d.ParticleType[n][i]-1]
            nearests = neighbor[(RIJ_norm-i_cutoffs) <= 0]
            CN  = nearests.shape[0] - 1
            nearests = nearests[RIJ_norm[nearests].argsort()]
            nearests = nearests[1:] + 1

            fneighbor.write('%d %d ' %(i + 1, CN))
            fneighbor.write(' '.join(map(str, nearests)))
            fneighbor.write('\n')

    fneighbor.close()
    print ('---Calculate nearest neighbors for atom-pairs done---')
