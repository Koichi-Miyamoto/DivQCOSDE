import numpy as np
import pandas as pd
from .MaximizeL import MaximizeLikelihood
from .QAEResult import QAEResult

def LowDepthQAE(a, epsilon, nShot, beta):

    theta = np.arcsin(np.sqrt(a))
    K = int(np.ceil(max(epsilon**(-2*beta), np.log(1/epsilon))))
    thetaMuls = []
    nShots = []
    n1s = []

    for k in range(1, K+1):
        nGrover = int(k**((1 - beta) / 2 / beta))
        thetaMul = 2 * nGrover + 1
        p = np.sin(thetaMul * theta)**2
        n1 = np.random.binomial(nShot, p)
        thetaMuls.append(thetaMul)
        nShots.append(nShot)
        n1s.append(n1)
    
    resDf = pd.DataFrame(dict(thetaMul=thetaMuls, nShot=nShots, n1=n1s))
    return QAEResult(MaximizeLikelihood(np.array(thetaMuls), np.array(nShots), np.array(n1s)), resDf)

