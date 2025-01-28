import numpy as np

class QAEResult:
    def __init__(self, aEst, measOutDf):
        self.aEst = aEst
        self.measOutDf = measOutDf
        self.TotalQueryNum = np.dot(measOutDf['thetaMul'], measOutDf['nShot'])
        self.MaxDepth = np.max(measOutDf['thetaMul'])