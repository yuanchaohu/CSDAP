#!/usr/bin/python
# coding = utf-8
#This module is part of an analysis package

Authorinfo = """
             ------------------Name: Yuan-Chao Hu--------------
             --------------Email: ychu0213@gmail.com-----------
             ----------Web: https://yuanchaohu.github.io/------
             """

docstr = """
         Calculate spherical harmonics of given (theta, phi) from (l = 0) to (l = 10)
         Formula taken from wikipedia 'https://en.wikipedia.org/wiki/Table_of_spherical_harmonics'

         From SphHarm0() to SphHarm10() a list of [-l, l] values will be returned

         if l > 10 use scipy.special.sph_harm (this may be slower)
         """

from math import pi, sin, cos, sqrt
import cmath

def SphHarm0(theta, phi):
    ''' Spherical Harmonics l = 0 '''
    results = []

    m0  = (1 / 2) * sqrt(1 / pi)
    results.append(m0)
    return results

def SphHarm1(theta, phi):
    ''' Spherical Harmonics l = 1 '''
    results = []

    mN1 = (1 / 2) * sqrt(3 / 2 / pi) * cmath.exp(-1j * phi) * sin(theta)
    results.append(mN1)
    m0  = (1 / 2) * sqrt(3 / pi) * cos(theta)
    results.append(m0)
    m1  = -(1 / 2) * sqrt(3 / 2 / pi) * cmath.exp(1j * phi) * sin(theta)
    results.append(m1)
    return results

def SphHarm2(theta, phi):
    ''' Spherical Harmonics l = 2 '''
    results = []

    mN2 =  (1 / 4) * sqrt(15 / 2 / pi) * cmath.exp(-2j * phi) * (sin(theta))**2
    results.append(mN2)
    mN1 =  (1 / 2) * sqrt(15 / 2 / pi) * cmath.exp(-1j *phi) * sin(theta) * cos(theta)
    results.append(mN1)
    m0  =  (1 / 4) * sqrt(5 / pi) * (3 * (cos(theta))**2 -1)
    results.append(m0)
    m1  = -(1 / 2) * sqrt(15 / 2 / pi) * cmath.exp(1j * phi) * sin(theta) * cos(theta)
    results.append(m1)
    m2  =  (1 / 4) * sqrt(15 / 2 / pi) * cmath.exp(2j * phi) * (sin(theta))**2
    results.append(m2)
    return results

def SphHarm3(theta, phi):
    ''' Spherical Harmonics l = 3 '''
    results = []

    mN3 =  (1 / 8) * sqrt(35 / pi) * cmath.exp(-3j * phi) * (sin(theta))**3
    results.append(mN3)
    mN2 =  (1 / 4) * sqrt(105 / 2 / pi) * cmath.exp(-2j * phi) * (sin(theta))**2 * cos(theta)
    results.append(mN2)
    mN1 =  (1 / 8) * sqrt(21 / pi) * cmath.exp(-1j * phi) * sin(theta) * (5 * (cos(theta))**2 - 1)
    results.append(mN1)
    m0  =  (1 / 4) * sqrt(7 / pi) * (5 * (cos(theta))**3 - 3 * cos(theta))
    results.append(m0)
    m1  = -(1 / 8) * sqrt(21 / pi) * cmath.exp(1j * phi) * sin(theta) * (5 * (cos(theta))**2 - 1)
    results.append(m1)
    m2  =  (1 / 4) * sqrt(105 / 2 / pi) * cmath.exp(2j * phi) * (sin(theta))**2 * cos(theta)
    results.append(m2)
    m3  = -(1 / 8) * sqrt(35 / pi) * cmath.exp(3j * phi) * (sin(theta))**3
    results.append(m3)
    return results

def SphHarm4(theta, phi):
    ''' Spherical Harmonics l = 4 '''
    results = []

    mN4 = (3 / 16) * sqrt(35 / 2 / pi) * cmath.exp(-4j * phi) * (sin(theta))**4
    results.append(mN4)
    mN3 = (3 / 8) * sqrt(35 / pi) * cmath.exp(-3j * phi) * (sin(theta))**3 * cos(theta)
    results.append(mN3)
    mN2 = (3 / 8) * sqrt(5 / 2 / pi) * cmath.exp(-2j * phi) * (sin(theta))**2 * (7 * (cos(theta))**2 - 1)
    results.append(mN2)
    mN1 = (3 / 8) * sqrt(5 / pi) * cmath.exp(-1j * phi) * sin(theta) * (7 * (cos(theta))**3 - 3 * cos(theta))
    results.append(mN1)
    m0  = (3 / 16) * sqrt(1 / pi) * (35 * (cos(theta))**4 - 30 * (cos(theta))**2 + 3)
    results.append(m0)
    m1  = -(3 / 8) * sqrt(5 / pi) * cmath.exp(1j * phi) * sin(theta) * (7 * (cos(theta))**3 - 3 * cos(theta))
    results.append(m1)
    m2  = (3 / 8) * sqrt(5 / 2 / pi) * cmath.exp(2j * phi) * (sin(theta))**2 * (7 * (cos(theta))**2 - 1)
    results.append(m2)
    m3  = -(3 / 8) * sqrt(35 / pi) * cmath.exp(3j * phi) * (sin(theta))**3 * cos(theta)
    results.append(m3)
    m4  = (3 / 16) * sqrt(35 / 2 / pi) * cmath.exp(4j * phi) * (sin(theta))**4
    results.append(m4)
    return results

def SphHarm5(theta, phi):
    ''' Spherical Harmonics l = 5 '''
    results = []

    mN5 = (3 / 32) * sqrt(77 / pi) * cmath.exp(-5j * phi) * (sin(theta))**5
    results.append(mN5)
    mN4 = (3 / 16) * sqrt(385 / 2 / pi) * cmath.exp(-4j * phi) * (sin(theta))**4 * cos(theta)
    results.append(mN4)
    mN3 = (1 / 32) * sqrt(385 / pi) * cmath.exp(-3j * phi) * (sin(theta))**3 * (9 * (cos(theta))**2 - 1)
    results.append(mN3)
    mN2 = (1 / 8) * sqrt(1155 / 2 / pi) * cmath.exp(-2j * phi) * (sin(theta))**2 * (3 * (cos(theta))**3 - cos(theta))
    results.append(mN2)
    mN1 = (1 / 16) * sqrt(165 / 2 / pi) * cmath.exp(-1j * phi) * sin(theta) * (21 * (cos(theta))**4 - 14 * (cos(theta))**2 + 1)
    results.append(mN1)
    m0  = (1 / 16) * sqrt(11 / pi) * (63 * (cos(theta))**5 - 70 * (cos(theta))**3 + 15 * cos(theta))
    results.append(m0)
    m1  = -(1 / 16) * sqrt(165 / 2 / pi) * cmath.exp(1j * phi) * sin(theta) * (21 * (cos(theta))**4 - 14 * (cos(theta))**2 + 1)
    results.append(m1)
    m2  = (1 / 8) * sqrt(1155 / 2 / pi) * cmath.exp(2j * phi) * (sin(theta))**2 * (3 * (cos(theta))**3 - cos(theta))
    results.append(m2)
    m3  = -(1 / 32) * sqrt(385 / pi) * cmath.exp(3j * phi) * (sin(theta))**3 * (9 * (cos(theta))**2 - 1)
    results.append(m3)
    m4  = (3 / 16) * sqrt(385 / 2 / pi) * cmath.exp(4j * phi) * (sin(theta))**4 * cos(theta)
    results.append(m4)
    m5  = -(3 / 32) * sqrt(77 / pi) * cmath.exp(5j * phi) * (sin(theta))**5 
    results.append(m5)
    return results 

def SphHarm6(theta, phi):
    ''' Spherical Harmonics l = 6 '''
    results = []

    mN6 = (1 / 64) * sqrt(3003 / pi) * cmath.exp(-6j * phi) * (sin(theta))**6
    results.append(mN6)
    mN5 = (3 / 32) * sqrt(1001 / pi) * cmath.exp(-5j * phi) * (sin(theta))**5 * cos(theta)
    results.append(mN5)
    mN4 = (3 / 32) * sqrt(91 / 2 / pi) * cmath.exp(-4j * phi) * (sin(theta))**4 * (11 * (cos(theta))**2 - 1)
    results.append(mN4)
    mN3 = (1 / 32) * sqrt(1365 / pi) * cmath.exp(-3j * phi) * (sin(theta))**3 * (11 * (cos(theta))**3 - 3 * cos(theta))
    results.append(mN3)
    mN2 = (1 / 64) * sqrt(1365 / pi) * cmath.exp(-2j * phi) * (sin(theta))**2 * (33 * (cos(theta))**4 - 18 * (cos(theta))**2 + 1)
    results.append(mN2)
    mN1 = (1 / 16) * sqrt(273 / 2 / pi) * cmath.exp(-1j * phi) * sin(theta) * (33 * (cos(theta))**5 - 30 * (cos(theta))**3 + 5 * cos(theta))
    results.append(mN1)
    m0  = (1 / 32) * sqrt(13 / pi) * (231 * (cos(theta))**6 - 315 * (cos(theta))**4 + 105 * (cos(theta))**2 - 5)
    results.append(m0)
    m1  = -(1 / 16) * sqrt(273 / 2 / pi) * cmath.exp(1j * phi) * sin(theta) * (33 * (cos(theta))**5 - 30 * (cos(theta))**3 + 5 * cos(theta))
    results.append(m1)
    m2  = (1 / 64) * sqrt(1365 / pi) * cmath.exp(2j * phi) * (sin(theta))**2 * (33 * (cos(theta))**4 - 18 * (cos(theta))**2 + 1)
    results.append(m2)
    m3  = -(1 / 32) * sqrt(1365 / pi) * cmath.exp(3j * phi) * (sin(theta))**3 * (11 * (cos(theta))**3 - 3 * cos(theta))
    results.append(m3)
    m4  = (3 / 32) * sqrt(91 / 2 / pi) * cmath.exp(4j * phi) * (sin(theta))**4 * (11 * (cos(theta))**2 - 1)
    results.append(m4)
    m5  = -(3 / 32) * sqrt(1001 / pi) * cmath.exp(5j * phi) * (sin(theta))**5 * cos(theta)
    results.append(m5)
    m6  = (1 / 64) * sqrt(3003 / pi) * cmath.exp(6j * phi) * (sin(theta))**6
    results.append(m6)
    return results

def SphHarm7(theta, phi):
    ''' Spherical Harmonics l = 7 '''
    results = []

    mN7 = (3 / 64) * sqrt(715 / 2 / pi) * cmath.exp(-7j * phi) * (sin(theta))**7
    results.append(mN7)
    mN6 = (3 / 64) * sqrt(5005 / pi) * cmath.exp(-6j * phi) * (sin(theta))**6 * cos(theta)
    results.append(mN6)
    mN5 = (3 / 64) * sqrt(385 / 2 / pi) * cmath.exp(-5j * phi) * (sin(theta))**5 * (13 * (cos(theta))**2 - 1)
    results.append(mN5)
    mN4 = (3 / 32) * sqrt( 385 / 2 /pi) * cmath.exp(-4j * phi) * (sin(theta))**4 * (13 * (cos(theta))**3 - 3 * cos(theta))
    results.append(mN4)
    mN3 = (3 / 64) * sqrt(35 / 2 / pi) * cmath.exp(-3j * phi) * (sin(theta))**3 * (143 * (cos(theta))**4 - 66 * (cos(theta))**2 + 3)
    results.append(mN3)
    mN2 = (3 / 64) * sqrt(35 / pi) * cmath.exp(-2j * phi) * (sin(theta))**2 * (143 * (cos(theta))**5 - 110 * (cos(theta))**3 + 15 * cos(theta))
    results.append(mN2)
    mN1 = (1 / 64) * sqrt(105 / 2 / pi) * cmath.exp(-1j * phi) * sin(theta) * (429 * (cos(theta))**6 - 495 * (cos(theta))**4 + 135 * (cos(theta))**2 - 5)
    results.append(mN1)
    m0  = (1 / 32) * sqrt(15 / pi) * (429 * (cos(theta))**7 - 693 * (cos(theta))**5 + 315 * (cos(theta))**3 - 35 * cos(theta))
    results.append(m0)
    m1  = -(1 / 64) * sqrt(105 / 2 / pi) * cmath.exp(1j * phi) * sin(theta) * (429 * (cos(theta))**6 - 495 * (cos(theta))**4 + 135 * (cos(theta))**2 - 5)
    results.append(m1)
    m2  = (3 / 64) * sqrt(35 / pi) * cmath.exp(2j * phi) * (sin(theta))**2 * (143 * (cos(theta))**5 - 110 * (cos(theta))**3 + 15 * cos(theta))
    results.append(m2)
    m3  = -(3 / 64) * sqrt(35 / 2 / pi) * cmath.exp(3j * phi) * (sin(theta))**3 * (143 * (cos(theta))**4 - 66 * (cos(theta))**2 + 3)
    results.append(m3)
    m4  = (3 / 32) * sqrt(385 / 2 / pi) * cmath.exp(4j * phi) * (sin(theta))**4 * (13 * (cos(theta))**3 - 3 * cos(theta))
    results.append(m4)
    m5  = -(3 / 64) * sqrt(385 / 2 / pi) * cmath.exp(5j * phi) * (sin(theta))**5 * (13 * (cos(theta))**2 - 1)
    results.append(m5)
    m6  = (3 / 64) * sqrt(5005 / pi) * cmath.exp(6j * phi) * (sin(theta))**6 * cos(theta)
    results.append(m6)
    m7  = -(3 / 64) * sqrt(715 / 2 / pi) * cmath.exp(7j * phi) * (sin(theta))**7 
    results.append(m7)
    return results

def SphHarm8(theta, phi):
    ''' Spherical Harmonics l = 8 '''
    results = []

    mN8 = (3 / 256) * sqrt(12155 / 2 / pi) * cmath.exp(-8j * phi) * (sin(theta))**8
    results.append(mN8)
    mN7 = (3 / 64) * sqrt(12155 / 2 / pi) * cmath.exp(-7j * phi) * (sin(theta))**7 * cos(theta)
    results.append(mN7)
    mN6 = (1 / 128) * sqrt(7293 / pi) * cmath.exp(-6j * phi) * (sin(theta))**6 * (15 * (cos(theta))**2 - 1)
    results.append(mN6)
    mN5 = (3 / 64) * sqrt(17017 / 2 / pi) * cmath.exp(-5j * phi) * (sin(theta))**5 * (5 * (cos(theta))**3 - cos(theta))
    results.append(mN5)
    mN4 = (3 / 128) * sqrt(1309 / 2 / pi) * cmath.exp(-4j * phi) * (sin(theta))**4 * (65 * (cos(theta))**4 - 26 * (cos(theta))**2 + 1)
    results.append(mN4)
    mN3 = (1 / 64) * sqrt(19635 / 2 / pi) * cmath.exp(-3j * phi) * (sin(theta))**3 * (39 * (cos(theta))**5 - 26 * (cos(theta))**3 + 3 * cos(theta))
    results.append(mN3)
    mN2 = (3 / 128) * sqrt(595 / pi) * cmath.exp(-2j * phi) * (sin(theta))**2 * (143 * (cos(theta))**6 - 143 * (cos(theta))**4 + 33 * (cos(theta))**2 - 1)
    results.append(mN2)
    mN1 = (3 / 64) * sqrt(17 / 2 / pi) * cmath.exp(-1j * phi) * sin(theta) * (715 * (cos(theta))**7 - 1001 * (cos(theta))**5 + 385 * (cos(theta))**3 - 35 * cos(theta))
    results.append(mN1)
    m0  = (1 / 256) * sqrt(17 / pi) * (6435 * (cos(theta))**8 - 12012 * (cos(theta))**6 + 6930 * (cos(theta))**4 - 1260 * (cos(theta))**2 + 35)
    results.append(m0)
    m1  = -(3 / 64) * sqrt(17 / 2 / pi) * cmath.exp(1j * phi) * sin(theta) * (715 * (cos(theta))**7 - 1001 * (cos(theta))**5 + 385 * (cos(theta))**3 - 35 * cos(theta))
    results.append(m1)
    m2  = (3 / 128) * sqrt(595 / pi) * cmath.exp(2j * phi) * (sin(theta))**2 * (143 * (cos(theta))**6 - 143 * (cos(theta))**4 + 33 * (cos(theta))**2 - 1)
    results.append(m2)
    m3  = -(1 / 64) * sqrt(19635 / 2 / pi) * cmath.exp(3j * phi) * (sin(theta))**3 * (39 * (cos(theta))**5 - 26 * (cos(theta))**3 + 3 * cos(theta))
    results.append(m3)
    m4  = (3 / 128) * sqrt(1309 / 2 / pi) * cmath.exp(4j * phi) * (sin(theta))**4 * (65 * (cos(theta))**4 - 26 * (cos(theta))**2 + 1)
    results.append(m4)
    m5  = -(3 / 64) * sqrt(17017 / 2 / pi) * cmath.exp(5j * phi) * (sin(theta))**5 * (5 * (cos(theta))**3 - cos(theta))
    results.append(m5)
    m6  = (1 / 128) * sqrt(7293 / pi) * cmath.exp(6j * phi) * (sin(theta))**6 * (15 * (cos(theta))**2 - 1)
    results.append(m6)
    m7  = -(3 / 64) * sqrt(12155 / 2 / pi) * cmath.exp(7j * phi) * (sin(theta))**7 * cos(theta)
    results.append(m7)
    m8  = (3 / 256) * sqrt(12155 / 2 / pi) * cmath.exp(8j * phi) * (sin(theta))**8
    results.append(m8)
    return results

def SphHarm9(theta, phi):
    ''' Spherical Harmonics l = 9 '''
    results = []

    mN9 = (1 / 512) * sqrt(230945 / pi) * cmath.exp(-9j * phi) * (sin(theta))**9
    results.append(mN9)
    mN8 = (3 / 256) * sqrt(230945 / 2 / pi) * cmath.exp(-8j * phi) * (sin(theta))**8 * cos(theta)
    results.append(mN8)
    mN7 = (3 / 512) * sqrt(13585 / pi) * cmath.exp(-7j * phi) * (sin(theta))**7 * (17 * (cos(theta))**2 - 1)
    results.append(mN7)
    mN6 = (1 / 128) * sqrt(40755 / pi) * cmath.exp(-6j * phi) * (sin(theta))**6 * (17 * (cos(theta))**3 - 3 * cos(theta))
    results.append(mN6)
    mN5 = (3 / 256) * sqrt(2717 / pi) * cmath.exp(-5j * phi) * (sin(theta))**5 * (85 * (cos(theta))**4 - 30 * (cos(theta))**2 + 1)
    results.append(mN5)
    mN4 = (3 / 128) * sqrt(95095 / 2 / pi) * cmath.exp(-4j * phi) * (sin(theta))**4 * (17 * (cos(theta))**5 - 10 * (cos(theta))**3 + cos(theta))
    results.append(mN4)
    mN3 = (1 / 256) * sqrt(21945 / pi) * cmath.exp(-3j * phi) * (sin(theta))**3 *(221 * (cos(theta))**6 - 195 * (cos(theta))**4 + 39 * (cos(theta))**2 - 1)
    results.append(mN3)
    mN2 = (3 / 128) * sqrt(1045 / pi) * cmath.exp(-2j * phi) * (sin(theta))**2 *( 221 * (cos(theta))**7 - 273 * (cos(theta))**5 + 91 * (cos(theta))**3 - 7* cos(theta))
    results.append(mN2)
    mN1 = (3 / 256) * sqrt(95 / 2 / pi) * cmath.exp(-1j * phi) * sin(theta) * (2431 * (cos(theta))**8 - 4004 * (cos(theta))**6 + 2002 * (cos(theta))**4 - 308 * (cos(theta))**2 +7)
    results.append(mN1)
    m0  = (1 / 256) * sqrt(19 / pi) * (12155 * (cos(theta))**9 - 25740 * (cos(theta))**7 + 18018 * (cos(theta))**5 - 4620 * (cos(theta))**3 + 315 * cos(theta))
    results.append(m0)
    m1  = -(3 / 256) * sqrt(95 / 2 / pi) * cmath.exp(1j * phi) * sin(theta) * (2431 * (cos(theta))**8 - 4004 * (cos(theta))**6 + 2002 * (cos(theta))**4 - 308 * (cos(theta))**2 + 7)
    results.append(m1)
    m2  = (3 / 128) * sqrt(1045 / pi) * cmath.exp(2j * phi) * (sin(theta))**2 * (221 * (cos(theta))**7 - 273 * (cos(theta))**5 + 91 * (cos(theta))**3 - 7 * cos(theta))
    results.append(m2)
    m3  = -(1 / 256) * sqrt(21945 / pi) * cmath.exp(3j * phi) * (sin(theta))**3 * (221 * (cos(theta))**6 - 195 * (cos(theta))**4 + 39 * (cos(theta))**2 - 1)
    results.append(m3)
    m4  = (3 / 128) * sqrt(95095 / 2 / pi) * cmath.exp(4j * phi) * (sin(theta))**4 * (17 * (cos(theta))**5 - 10 * (cos(theta))**3 + cos(theta))
    results.append(m4)
    m5  = -(3 / 256) * sqrt(2717 / pi) * cmath.exp(5j * phi) * (sin(theta))**5 * (85 * (cos(theta))**4 - 30 * (cos(theta))**2 + 1)
    results.append(m5)
    m6  = (1 / 128) * sqrt(40755 / pi) * cmath.exp(6j * phi) * (sin(theta))**6 * (17 * (cos(theta))**3 - 3 * cos(theta))
    results.append(m6)
    m7  = -(3 / 512) * sqrt(13585 / pi) * cmath.exp(7j * phi) * (sin(theta))**7 * (17 * (cos(theta))**2 - 1)
    results.append(m7)
    m8  = (3 / 256) * sqrt(230945 / 2 / pi) * cmath.exp(8j * phi) * (sin(theta))**8 * cos(theta)
    results.append(m8)
    m9  = -(1 / 512) * sqrt(230945 / pi) * cmath.exp(9j * phi) * (sin(theta))**9
    results.append(m9)
    return results

def SphHarm10(theta, phi):
    ''' Spherical Harmonics l = 10 '''
    results = []

    mN10 = (1 / 1024) * sqrt(969969 / pi) * cmath.exp(-10j * phi) * (sin(theta))**10
    results.append(mN10)
    mN9  = (1 / 512) * sqrt(4849845 / pi) * cmath.exp(-9j * phi) * (sin(theta))**9 * cos(theta)
    results.append(mN9)
    mN8  = (1 / 512) * sqrt(255255 / 2 / pi) * cmath.exp(-8j * phi) * (sin(theta))**8 * (19 * (cos(theta))**2 - 1)
    results.append(mN8)
    mN7  = (3 / 512) * sqrt(85085 / pi) * cmath.exp(-7j * phi) * (sin(theta))**7 * (19 * (cos(theta))**3 - 3 * cos(theta))
    results.append(mN7)
    mN6  = (3 / 1024) * sqrt(5005 / pi) * cmath.exp(-6j * phi) * (sin(theta))**6 * (323 * (cos(theta))**4 - 102 * (cos(theta))**2 + 3)
    results.append(mN6)
    mN5  = (3 / 256) * sqrt(1001 / pi) * cmath.exp(-5j * phi) * (sin(theta))**5 * (323 * (cos(theta))**5 - 170 * (cos(theta))**3 + 15 * cos(theta))
    results.append(mN5)
    mN4  = (3 / 256) * sqrt(5005 / 2 / pi) * cmath.exp(-4j * phi) * (sin(theta))**4 * (323 * (cos(theta))**6 - 255 * (cos(theta))**4 + 45 * (cos(theta))**2 - 1)
    results.append(mN4)
    mN3  = (3 / 256) * sqrt(5005 / pi) * cmath.exp(-3j * phi) * (sin(theta))**3 * (323 * (cos(theta))**7 - 357 * (cos(theta))**5 + 105 * (cos(theta))**3 - 7 * cos(theta))
    results.append(mN3)
    mN2  = (3 / 512) * sqrt(385 / 2 / pi) * cmath.exp(-2j * phi) * (sin(theta))**2 * (4199 * (cos(theta))**8 - 6188 * (cos(theta))**6 + 2730 * (cos(theta))**4 - 364 * (cos(theta))**2 + 7)
    results.append(mN2)
    mN1  = (1 / 256) * sqrt(1155 / 2 / pi) * cmath.exp(-1j * phi) * sin(theta) * (4199 * (cos(theta))**9 - 7956 * (cos(theta))**7 + 4914 * (cos(theta))**5 - 1092 * (cos(theta))**3 + 63 * cos(theta))
    results.append(mN1)
    m0   = (1 / 512) * sqrt(21 / pi) * (46189 * (cos(theta))**10 - 109395 * (cos(theta))**8 + 90090 * (cos(theta))**6 - 30030 * (cos(theta))**4 + 3465 * (cos(theta))**2 - 63)
    results.append(m0)
    m1   = -(1 / 256) * sqrt(1155 / 2 / pi) * cmath.exp(1j * phi) * sin(theta) * (4199 * (cos(theta))**9 - 7956 * (cos(theta))**7 + 4914 * (cos(theta))**5 - 1092 * (cos(theta))**3 + 63 * cos(theta))
    results.append(m1)
    m2   = (3 / 512) * sqrt(385 / 2 / pi) * cmath.exp(2j * phi) * (sin(theta))**2 * (4199 * (cos(theta))**8 - 6188 * (cos(theta))**6 + 2730 * (cos(theta))**4 - 364 * (cos(theta))**2 + 7)
    results.append(m2)
    m3   = -(3 / 256) * sqrt(5005 / pi) * cmath.exp(3j * phi) * (sin(theta))**3 * (323 * (cos(theta))**7 - 357 * (cos(theta))**5 + 105 * (cos(theta))**3 - 7 * cos(theta))
    results.append(m3)
    m4   = (3 / 256) * sqrt(5005 / 2 / pi) * cmath.exp(4j * phi) * (sin(theta))**4 * (323 * (cos(theta))**6 - 255 * (cos(theta))**4 + 45 * (cos(theta))**2 - 1)
    results.append(m4)
    m5   = -(3 / 256) * sqrt(1001 / pi) * cmath.exp(5j * phi) * (sin(theta))**5 * (323 * (cos(theta))**5 - 170 * (cos(theta))**3 + 15 * cos(theta))
    results.append(m5)
    m6   = (3 / 1024) * sqrt(5005 / pi) * cmath.exp(6j * phi) * (sin(theta))**6 * (323 * (cos(theta))**4 - 102 * (cos(theta))**2 + 3)
    results.append(m6)
    m7   = -(3 / 512) * sqrt(85085 / pi) * cmath.exp(7j * phi) * (sin(theta))**7 * (19 * (cos(theta))**3 - 3 * cos(theta))
    results.append(m7)
    m8   = (1 / 512) * sqrt(255255 / 2 / pi) * cmath.exp(8j * phi) * (sin(theta))**8 * (19 * (cos(theta))**2 - 1)
    results.append(m8)
    m9   = -(1 / 512) * sqrt(4849845 / pi) * cmath.exp(9j * phi) * (sin(theta))**9 * cos(theta)
    results.append(m9)
    m10  = (1 / 1024) * sqrt(969969 / pi) * cmath.exp(10j * phi) * (sin(theta))**10
    results.append(m10)
    return results 


def SphHarm_above(l, theta, phi):
    ''' Spherical Harmonics l > 10 '''
    from scipy.special import sph_harm
    #be aware of theta and phi used for scipy is inverse to the above equations

    #change phi from [-PI, PI] to [0, 2PI]
    if phi < 0: phi += 2 * pi
    
    results = []
    for m in range(-l, l + 1):
        results.append(sph_harm(m, l, phi, theta))
    return results