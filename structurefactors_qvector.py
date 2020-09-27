#!/usr/bin/python
# coding = utf-8
#This module is part of an analysis package

Authorinfo = """
             ------------------Name: Yuan-Chao Hu--------------
             --------------Email: ychu0213@gmail.com-----------
             ----------The University of Tokyo (2020)----------
             """

Docstr = """
         This module calculates total/partial structure factors
         for a single wavevector
         
         The box needs to be orthogonal but not necessarily cubic
         in case the box lengths in different dimensions are provided
         as a numpy array [Lx, Ly, Lz]
         """

import os
import numpy  as np 
from math import pi, sin, cos


def Sq_total(qvector, Positions, Boxlength):
    """calculate total S(q) for a single wavevector q
    
    qvector is set as [1, 1, 0] for example
    Positions should be a list of atomic positions
    Boxlength prefers to be a numpy array for all dimensions
    """

    twopidl = 2.0 * pi / Boxlength
    qvector = twopidl * qvector
    qvalue = np.linalg.norm(qvector)
    numofconfig = len(Positions)
    numofatoms = Positions[0].shape[0]

    results = 0
    for n in range(numofconfig):
        sin_part = 0
        cos_part = 0
        for i in range(numofatoms):
            theta = np.dot(qvector, Positions[n][i])
            sin_part += sin(theta)
            cos_part += cos(theta)

        results += sin_part**2 + cos_part**2
    results = results / numofconfig / numofatoms

    return qvalue, results

def Sq_partial(alpha, beta, qvector, Positions, Boxlength, ParticleType, ndim=2):
    """Calculate partial S(q) for atom types alpha and beta for a single wavevector
    
    qvector is set as [1, 1, 0] for example
    Positions should be a list of atomic positions including all particles
    Boxlength prefers to be a numpy array for all dimensions
    ParticleType should be a numpy array including all particles
    """

    twopidl = 2.0 * pi / Boxlength
    qvector = twopidl * qvector
    qvalue  = np.linalg.norm(qvector)
    numofconfig = len(Positions)
    numofatoms  = ParticleType.shape[0]

    type_alpha = ParticleType == alpha
    N_alpha = type_alpha.sum()
    type_beta = ParticleType == beta
    N_beta = type_beta.sum()

    results = 0
    for n in range(numofconfig):
        thetas = Positions[n][:, 0]*qvector[0] + Positions[n][:, 1]*qvector[1]
        if ndim==3:
            thetas += Positions[n][:, 2]*qvector[2]
        
        sin_part = np.sin(thetas)
        cos_part = np.cos(thetas)

        results += sin_part[type_alpha].sum() * sin_part[type_beta].sum()
        results += cos_part[type_alpha].sum() * cos_part[type_beta].sum()

    results = results / numofconfig / np.sqrt(N_alpha * N_beta)

    return qvalue, results