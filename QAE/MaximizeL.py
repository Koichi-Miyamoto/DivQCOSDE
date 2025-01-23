import numpy as np
import pandas as pd
import scipy as sp

def MaximizeLikelihood(thetaMuls, nShots, n1s, method='L-BFGS-B'):

    n0s = nShots - n1s
    thetaMulMax = np.max(thetaMuls)
    thetaInis = np.linspace(0, 0.5 * np.pi, int(thetaMulMax)+1)

    neglogL = lambda theta: -np.dot(n1s, np.log(np.sin(thetaMuls * theta)**2)) - np.dot(n0s, np.log(np.cos(thetaMuls * theta)**2))
    dneglogL = lambda theta: -np.dot(n1s, 2 * thetaMuls / np.tan(thetaMuls * theta)) + np.dot(n0s, -2 * thetaMuls * np.tan(thetaMuls * theta))

    thetaOpt = np.nan
    neglogLOpt = np.Inf
    for i in range(len(thetaInis)-1):
        thetaIni = 0.5 * (thetaInis[i] + thetaInis[i+1])
        minimizeRes = sp.optimize.minimize(neglogL, thetaIni, method=method, jac=dneglogL, bounds=[[thetaInis[i], thetaInis[i+1]]])
        thetaOptTemp = minimizeRes.x[0]
        neglogLOptTemp = neglogL(thetaOptTemp)
        if neglogLOptTemp < neglogLOpt:
            thetaOpt = thetaOptTemp
            neglogLOpt = neglogLOptTemp

    return np.sin(thetaOpt)**2

