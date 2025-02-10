import numpy as np
import scipy as sp

class LegExpResult:
    def __init__(self, coefs):
        self.coefs = coefs
        fApp = float(coefs[0]) * sp.special.legendre(0)
        for l in range(1, len(coefs)):
            fApp = fApp + float(coefs[l]) * sp.special.legendre(l)
        self.fApp = fApp

def LegExp(f, maxDeg):

    coefs = np.zeros(maxDeg+1)
    coefs[0] = 0.5 # assume that f is a PDF on [-1,1]

    for l in range(1, maxDeg+1):
        legPoly = sp.special.legendre(l)
        integFunc = lambda x: legPoly(x) * f(x)
        integ, _ = sp.integrate.quad(integFunc, -1, 1)
        coefs[l] = (l + 0.5) * integ

    return LegExpResult(coefs)

def LegExpSPDensity(densFunc0, transDensFunc, maxDeg, lbx=-1, ubx=1, epsabs=1.49e-8):

    coefs = np.zeros(maxDeg+1)
    coefs[0] = 0.5 # assume that f is a PDF on [-1,1]

    for l in range(1, maxDeg+1):
        legPoly = sp.special.legendre(l)
        integFunc = lambda x, x0: legPoly(x) * densFunc0(x0) * transDensFunc(x, x0)
        integ, _ = sp.integrate.dblquad(integFunc, -1, 1, lbx, ubx, epsabs=epsabs)
        coefs[l] = (l + 0.5) * integ

    return LegExpResult(coefs)