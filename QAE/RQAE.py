import numpy as np
import pandas as pd
from .MaximizeL import MaximizeLikelihood
from .QAEResult import QAEResult

def RQAE(a, K, R, method='L-BFGS-B'):

    theta = np.arcsin(np.sqrt(a))
    thetaMuls = []
    nShots = []
    n1s = []

    for i in range(1, K+1):
        if i == 1:
            n1 = np.random.binomial(R, a)
            thetaMuls.append(1)
            nShots.append(R)
            n1s.append(n1)
        else:
            for nOra in np.random.randint(2**(i-1), 2**i, size=R):
                n1 = np.random.binomial(1, np.sin(nOra * theta)**2)
                thetaMuls.append(nOra)
                nShots.append(1)
                n1s.append(n1)

    resDf = pd.DataFrame(dict(thetaMul=thetaMuls, nShot=nShots, n1=n1s))
    return QAEResult(MaximizeLikelihood(np.array(thetaMuls), np.array(nShots), np.array(n1s), method=method), resDf)

