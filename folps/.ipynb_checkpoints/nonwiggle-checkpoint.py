import warnings
import numpy as np
from scipy import interpolate
from scipy import special


from scipy.fft import dst, idst

from scipy.interpolate import CubicSpline
from scipy import interpolate
from scipy.fft import dst, idst
from scipy.special import gamma
from scipy.special import spherical_jn
from scipy.special import eval_legendre
from scipy.integrate import quad
import sys



def interp(k, x, y):
    '''Cubic spline interpolation.
    
    Args:
        k: coordinates at which to evaluate the interpolated values.
        x: x-coordinates of the data points.
        y: y-coordinates of the data points.
    Returns:
        Cubic interpolation of ‘y’ evaluated at ‘k’.
    '''
    inter = CubicSpline(x, y)
    return inter(k) 

def pknwJ(k, PSLk, h):
    '''Routine (based on J. Hamann et. al. 2010, arXiv:1003.3999) to get the non-wiggle piece of the linear power spectrum.    
    
    Args:
        k: wave-number.
        PSLk: linear power spectrum.
        h: H0/100.
    Returns:
        non-wiggle piece of the linear power spectrum.
    '''
    #ksmin(max): k-range and Nks: points
    ksmin = 7*10**(-5)/h; ksmax = 7/h; Nks = 2**16

    #sample ln(kP_L(k)) in Nks points, k range (equidistant)
    ksT = [ksmin + ii*(ksmax-ksmin)/(Nks-1) for ii in range(Nks)]
    PSL = interp(ksT, k, PSLk)
    logkpkT = np.log(ksT*PSL)
        
    #Discrete sine transf., check documentation
    FSTtype = 1; m = int(len(ksT)/2)
    FSTlogkpkT = dst(logkpkT, type = FSTtype, norm = "ortho")
    FSTlogkpkOddT = FSTlogkpkT[::2]
    FSTlogkpkEvenT = FSTlogkpkT[1::2]
        
    #cut range (remove the harmonics around BAO peak)
    mcutmin = 120; mcutmax = 240;
        
    #Even
    xEvenTcutmin = np.linspace(1, mcutmin-2, mcutmin-2)
    xEvenTcutmax = np.linspace(mcutmax+2, len(FSTlogkpkEvenT), len(FSTlogkpkEvenT)-mcutmax-1)
    EvenTcutmin = FSTlogkpkEvenT[0:mcutmin-2] 
    EvenTcutmax = FSTlogkpkEvenT[mcutmax+1:len(FSTlogkpkEvenT)]
    xEvenTcuttedT = np.concatenate((xEvenTcutmin, xEvenTcutmax))
    nFSTlogkpkEvenTcuttedT = np.concatenate((EvenTcutmin, EvenTcutmax))


    #Odd
    xOddTcutmin = np.linspace(1, mcutmin-1, mcutmin-1)
    xOddTcutmax = np.linspace(mcutmax+1, len(FSTlogkpkEvenT), len(FSTlogkpkEvenT)-mcutmax)
    OddTcutmin = FSTlogkpkOddT[0:mcutmin-1]
    OddTcutmax = FSTlogkpkOddT[mcutmax:len(FSTlogkpkEvenT)]
    xOddTcuttedT = np.concatenate((xOddTcutmin, xOddTcutmax))
    nFSTlogkpkOddTcuttedT = np.concatenate((OddTcutmin, OddTcutmax))

    #Interpolate the FST harmonics in the BAO range
    preT, = map(np.zeros,(len(FSTlogkpkT),))
    PreEvenT = interp(np.linspace(2, mcutmax, mcutmax-1), xEvenTcuttedT, nFSTlogkpkEvenTcuttedT)
    PreOddT = interp(np.linspace(0, mcutmax-2, mcutmax-1), xOddTcuttedT, nFSTlogkpkOddTcuttedT)
    for ii in range(m):
        if (mcutmin < ii+1 < mcutmax):
            preT[2*ii+1] = PreEvenT[ii]
            preT[2*ii] = PreOddT[ii]
        if (mcutmin >= ii+1 or mcutmax <= ii+1):
            preT[2*ii+1] = FSTlogkpkT[2*ii+1]
            preT[2*ii] = FSTlogkpkT[2*ii]
                
        
    #Inverse Sine transf.
    FSTofFSTlogkpkNWT = idst(preT, type = FSTtype, norm = "ortho")
    PNWT = np.exp(FSTofFSTlogkpkNWT)/ksT

    PNWk = interp(k, ksT, PNWT)
    DeltaAppf = k*(PSL[7]-PNWT[7])/PNWT[7]/ksT[7]

    irange1 = np.where((k < 1e-3))
    PNWk1 = PSLk[irange1]/(DeltaAppf[irange1] + 1)

    irange2 = np.where((1e-3 <= k) & (k <= ksT[len(ksT)-1]))
    PNWk2 = PNWk[irange2]
        
    irange3 = np.where((k > ksT[len(ksT)-1]))
    PNWk3 = PSLk[irange3]
        
    PNWkTot = np.concatenate((PNWk1, PNWk2, PNWk3))
        
    return(k, PNWkTot)


