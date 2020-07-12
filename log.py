#coding = utf-8

import pandas as pd 
import numpy as np 
import subprocess

def lammpslog(filename):
    """extract the thermodynamic quantities from lammp log file"""

    with open(filename, 'r') as f:
        data = f.readlines()

    #----get how many sections are there----
    start = [i for i, val in enumerate(data) if val.startswith('Step ')]
    end   = [i for i, val in enumerate(data) if val.startswith('Loop time of ')]

    if data[-1] is not '\n':
        if data[-1].split()[0].isnumeric(): #incomplete log file
            end.append(len(data) - 2)
    
    start   = np.array(start)
    end     = np.array(end)
    linenum = end - start - 1
    print ('Section Number: %d' %len(linenum), '    Line Numbers: ' + str(linenum))
    del data 

    final = []
    for i in range(len(linenum)):
        data = pd.read_csv(filename, sep = '\s+', skiprows = start[i], nrows = linenum[i])
        final.append(data)
        del data

    return final

def hoomdlog(filename):
    """extract the thermodynamic quantities from hoomd log file"""

    data = pd.read_csv(filename, sep = '\s+')
    return data