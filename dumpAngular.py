#!/usr/bin/python
#coding = utf-8
#This module is part of an analysis package

Authorinfo = """
             ------------------Name: Yuan-Chao Hu--------------
             --------------Email: ychu0213@gmail.com-----------
             ----------Web: https://yuanchaohu.github.io/------
             """

Docstr = """
        This module is used to read the angular vectors of particles
        for the calculation of rotational dynamics

        It can get vectors from patchy particles by knowing the positions
        of particles and patches, the first patch will be used as 
        reference to define the vectors
"""

import numpy as np 
from RewriteDump import lammps
from dump import readdump

class readangular():
    """read angular velocity from simulations"""

    def __init__(self, filename, ndim = 3, *arg):
        self.filename   = filename
        self.ndim       = ndim
        self.TimeStep   = []
        self.ParticleNumber = []
        self.ParticleType   = []
        self.velocity       = []
        self.SnapshotNumber = 0

    def read_onefile(self):
        f = open(self.filename, 'r')
        velocity = self.read_configs(f)
        while velocity.any():
            self.velocity.append(velocity)
            velocity = self.read_configs(f)

        f.close()
        self.SnapshotNumber = len(self.TimeStep)
        print ('----------Angular vector reading over----------')

    def read_configs(self, f):
        """Read one configuration a time"""
        try:
            item = f.readline()
            timestep = int(item.split()[1])
            self.TimeStep.append(timestep)
            item = f.readline()
            ParticleNumber = int(item.split()[1])
            self.ParticleNumber.append(ParticleNumber)

            f.readline()
            #-----------------------------
            ParticleType = np.zeros(ParticleNumber, dtype = np.int)
            velocity = np.zeros((ParticleNumber, self.ndim))
            for i in range(ParticleNumber):
                item = f.readline().split()
                atomid = int(item[0]) - 1
                ParticleType[atomid] = int(item[1])
                velocity[atomid] = [float(j) for j in item[2: 2 + self.ndim]]

            self.ParticleType.append(ParticleType)
            return velocity
        except:
            velocity = np.zeros((self.ParticleNumber[0], self.ndim))
            return velocity


def PatchVector(filename, num_patch = 12, ndim = 3, filetype = 'lammps', outputvec = '', outputdump = ''):
    """get the vector of each particle based on its patches


    Two files will be generated:
    1) patch-particle vector for each particle
    2) coordinates of each particle and its first patch as LAMMPS dump format
    """

    #-----get the first patch of each particle-----
    f = open(filename, 'r')
    for i in range(3):
        f.readline()
    num_total = int(f.readline())
    for i in range(5):
        f.readline()

    num_atom = int(num_total / (num_patch+1))
    print ('Atom Number: %d' %num_atom)
    hosts = []
    firstpatch = []
    for i in range(num_atom):
        hosts.append(int(f.readline().split()[0]))
        firstpatch.append(int(f.readline().split()[0]))
        for j in range(num_patch - 1):
            f.readline()
    f.close()

    if hosts[-1] + 1 != firstpatch[0]:
        print ('Warning: possible ERROR about particle-patch relation')
    
    #-----get positions------
    hosts      = np.array(hosts) - 1
    firstpatch = np.array(firstpatch) - 1
    d = readdump(filename, ndim, filetype)
    d.read_onefile()

    pos_centers = [i[hosts] for i in d.Positions]
    pos_patches = [i[firstpatch] for i in d.Positions]

    #------move particles inside of box (PBC)--------
    for n in range(d.SnapshotNumber):
        halfbox = d.Boxlength[n].min() / 2.0
        for i in range(num_atom):
            RIJ  = pos_centers[n][i] - pos_patches[n][i]
            periodic = np.where(np.abs(RIJ) > halfbox, np.sign(RIJ), 0).astype(np.int)
            pos_patches[n][i] += periodic * d.Boxlength[n]

    #-----calculate vectors------------------
    vectors = []
    for n in range(d.SnapshotNumber):
        vectors.append(pos_patches[n] - pos_centers[n])

    print ('-------Particle-Patch vector done------')
    
    #-------output information on vectors--------------
    fvec = open(outputvec, 'w')
    for n in range(d.SnapshotNumber):
        fvec.write('Step %d\n' %d.TimeStep[n])
        fvec.write('Natom %d\n' %num_atom)
        fvec.write('id type vx vy vz\n')
        for j, i in enumerate(hosts):
            fvec.write('%d %d '%(j+1, d.ParticleType[n][i]))
            fvec.write('%.6f %.6f %.6f\n'%tuple(vectors[n][j]))
    fvec.close()

    #-------output information on paritcle-patch-------
    if outputdump:
        fdump = open(outputdump, 'w')
        totalnum = num_atom + len(firstpatch)
        for n in range(d.SnapshotNumber):
            header = lammps(d.TimeStep[n], totalnum, d.Boxbounds[n])
            fdump.write(header)
            for j, i in enumerate(hosts):
                fdump.write('%d %d '%(j+1, d.ParticleType[n][i]))
                fdump.write('%.6f %.6f %.6f\n'%tuple(d.Positions[n][i]))

            for j, i in enumerate(firstpatch):
                fdump.write('%d %d '%(num_atom + j + 1, d.ParticleType[n][i]))
                fdump.write('%.6f %.6f %.6f\n'%tuple(d.Positions[n][i]))
        fdump.close()

    print ('--------wrting configuration information over--------')
    return None