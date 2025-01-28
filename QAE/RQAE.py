import numpy as np
import pandas as pd
from .MaximizeL import MaximizeLikelihood
from .QAEResult import QAEResult

def RQAE(a, epsilon, R):

    theta = np.arcsin(np.sqrt(a))
    thetaMuls = []
    nShots = []
    n1s = []
    intEpsInv = int(np.ceil(1 / epsilon))
    K = int(np.log2(1 / epsilon))
    nRound = K + 1 if 2**K < intEpsInv else K

    for i in range(nRound):
        if i == 0:
            n1 = np.random.binomial(R, a)
            thetaMuls.append(1)
            nShots.append(R)
            n1s.append(n1)
        else:
            maxM = intEpsInv if i == K else 2**(i+1)
            for nOra in np.random.randint(2**i, maxM, size=R):
                n1 = np.random.binomial(1, np.sin(nOra * theta)**2)
                thetaMuls.append(nOra)
                nShots.append(1)
                n1s.append(n1)

    resDf = pd.DataFrame(dict(thetaMul=thetaMuls, nShot=nShots, n1=n1s))
    return QAEResult(MaximizeLikelihood(np.array(thetaMuls), np.array(nShots), np.array(n1s)), resDf)

