#############################################
##          fortytwo_util_prod.py          ##
#     Functions for increased productivity  #
#############################################

####################################################################
import sys
import math
import numpy as np
from astropy.constants import c
from astropy.constants import M_sun
from astropy.constants import G
from astropy.constants import au
from astropy.coordinates import SkyCoord
from astropy import units
Msun = M_sun.value
c = c.value
au=au.value
G=G.value
####################################################################

def getargv(argv, key):
    for i in range(0,len(argv)):
        arg=argv[i]
        if (arg==key):
            return argv[i+1]

def chkargv(argv, key):
    for i in range(0,len(argv)):
        arg=argv[i]
        if (arg==key):
            return True

    return False

def chk_range(vpar, par_range):
    if len(par_range)>1:
        bola=vpar>=par_range[:,0]
        bolb=vpar<=par_range[:,1]
        bola=np.logical_not(np.logical_and(bola,bolb))
        return np.logical_not(np.any(bola))
    else:
        return True;

def cvtRAStrtoRad(strRA):
    h, m, s = np.float64(tuple(np.array(strRA.split(':'))))
    return (h + m / 60. + s / 3600.) / 12. * np.pi

def cvtDECStrtoRad(strDec):
    d, m, s = np.float64(tuple(np.array(strDec.split(':'))))
    return (np.abs(d) + m / 60. + s / 3600.) / 180. * np.pi * np.sign(d)

def appdiag(f1, f2):
    n1, m1 = f1.shape
    n2, m2 = f2.shape
    if n1 == 0 or m1 == 0:
        fall = f2
    else:
        fall = np.hstack((np.vstack((f1, np.zeros((n2, m1)))), np.vstack((np.zeros((n1, m2)), f2))))
    return fall

class DualOutput:
    def __init__(self, filename):
        self.stdout = sys.stdout
        self.file = open(filename, 'w')

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        # Ensure that the output is seen immediately
        self.stdout.flush()
        self.file.flush()

def check_matrix_posdef(A):
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False
