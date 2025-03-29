import numpy as np
from sklearn.decomposition import PCA

from Global_vars import Global_vars
from Model_ALSTM_DTCN import Model_ALSTM_DTCN


def objfun_Feat(Soln):
    pca = PCA(n_components=1)
    Fitn = np.zeros(Soln.shape[0])
    trainX = Global_vars.trainData
    testX = Global_vars.testData
    trainY = Global_vars.trainTarget
    testY = Global_vars.testTarget
    Data = Global_vars.Data
    Feats = Global_vars.Feat
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = Soln[i, :]
            feat1 = Data[:, sol[:10].astype(np.int)] * sol[10:20]
            feat2 = Data[:, sol[20:30].astype(np.int)] * sol[30:40]
            feat3 = []
            for i in range(Data.shape[0]):
                data = Data[i, :]
                X_train = pca.fit_transform(data.reshape(-1, 1))
                X_test = pca.transform(data.reshape(-1, 1)).reshape(-1)[:10]
                feat3.append(X_test)
            feat3 = np.asarray(feat3)
            feat = sol[40] * feat1 + (1 - sol[40]) * feat2
            feat = sol[41] * feat + (1 - sol[41]) * feat3
            Fitn[i] = 1 / np.mean(np.corrcoef(feat))
        return Fitn
    else:
        sol = Soln
        feat1 = Data[:, sol[:10].astype(np.int)] * sol[10:20]
        feat2 = Data[:, sol[20:30].astype(np.int)] * sol[30:40]
        feat3 = []
        for i in range(Data.shape[0]):
            data = Data[i, :]
            X_train = pca.fit_transform(data.reshape(-1, 1))
            X_test = pca.transform(data.reshape(-1, 1)).reshape(-1)[:10]
            feat3.append(X_test)
        feat3 = np.asarray(feat3)
        feat = sol[40] * feat1 + (1 - sol[40]) * feat2
        feat = sol[41] * feat + (1 - sol[41]) * feat3
        Fit = 1 / np.mean(np.corrcoef(feat))
        return Fit


def objfun_LSTM(Soln):
    Fitn = np.zeros(Soln.shape[0])
    trainX = Global_vars.trainData
    testX = Global_vars.testData
    trainY = Global_vars.trainTarget
    testY = Global_vars.testTarget
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = Soln[i, :]
            Eval, pred = Model_ALSTM_DTCN(trainX, trainY, testX, testY, sol)
            Fitn[i] = 1 / (Eval[4] + Eval[7])
        return Fitn
    else:
        sol = Soln
        Eval, pred = Model_ALSTM_DTCN(trainX, trainY, testX, testY, sol)
        Fit = 1 / (Eval[4] + Eval[7])
        return Fit
