import csv

from numpy import matlib

from AOA import AOA
from CO import CO
from DPO import DPO
from HHO import HHO
from Model_1DCNN import Model_1DCNN
from Model_AttentionLSTM import Model_AttentionLSTM
from Model_DTCN import Model_DTCN
from Model_LSTM import Model_LSTM
from ObjectiveFunction import *
from Plot_Results import *
from Proposed import Proposed


def Read_Dataset(filename, tarInd=3, startInd=0):
    count = 0
    Data = []
    with open(filename, mode='r') as file:
        csvFile = csv.reader(file)
        for lines in csvFile:
            if count >= 1:
                Data.append(np.asarray(lines[startInd:]).astype(float))
            count += 1
    isCovid = np.random.randint(low=0, high=2, size=(len(Data), 1))
    Data = np.asarray(Data)
    Target = Data[:, tarInd].reshape(-1, 1)
    Data = np.append(Data, isCovid, axis=1)
    Min = np.min(Data[:, 1]) * 0.1
    Max = np.max(Data[:, 1]) * 0.1
    addorsub = np.random.randint(low=0, high=2, size=Data.shape[0])
    variation = Min + (Max - Min) * np.random.random((Data.shape[0], 1))
    Target[addorsub == 0] = Target[addorsub == 0] + variation[addorsub == 0]
    Target[addorsub == 1] = Target[addorsub == 1] - variation[addorsub == 1]
    return Data, Target


noOfDataset = 3

# Read Dataset
an = 0
if an:
    Data, Target = Read_Dataset('./Dataset/Dataset 1/diabetes.csv', tarInd=1)
    np.save('Data_1.npy', Data)
    np.save('Target_1.npy', Target)

    Data, Target = Read_Dataset('./Dataset/Dataset 2/heart.csv', tarInd=4)
    np.save('Data_2.npy', Data)
    np.save('Target_2.npy', Target)

    Data, Target = Read_Dataset('./Dataset/Dataset 3/lung_cancer_examples.csv', startInd=2)
    np.save('Data_3.npy', Data)
    np.save('Target_3.npy', Target)

# Optimization for Feature Selection in 1DCNN
an = 0
if an:
    Bestsol = []
    for i in range(noOfDataset):
        Data = np.load('Data_' + str(i + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(i + 1) + '.npy', allow_pickle=True)

        learnperc = round(Data.shape[0] * 0.75)
        Global_vars.trainData = Data[:learnperc, :]
        Global_vars.testData = Data[learnperc:, :]
        Global_vars.trainTarget = Target[:learnperc, :]
        Global_vars.testTarget = Target[learnperc:, :]

        Feats, pred = Model_1DCNN(Global_vars.trainData, Global_vars.trainTarget, Global_vars.testData,
                                  Global_vars.testTarget)
        Global_vars.Data = Data
        Global_vars.Feat = Feats

        Npop = 10
        Chlen = 4
        xmin = matlib.repmat(np.zeros(42), Npop, 1)
        xmax = matlib.repmat(np.append((Feats.shape[1] - 1) * np.ones(10), np.append(np.ones(10),
                                                                                     np.append(
                                                                                         (Data.shape[1] - 1) * np.ones(
                                                                                             10),
                                                                                         np.append(np.ones(10),
                                                                                                   np.ones(2))))), Npop,
                             1)
        initsol = np.zeros(xmax.shape)
        for p1 in range(Npop):
            for p2 in range(xmax.shape[1]):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        fname = objfun_Feat
        Max_iter = 25

        print("HHO...")
        [bestfit1, fitness1, bestsol1, time1] = HHO(initsol, fname, xmin, xmax, Max_iter)

        print("AOA...")
        [bestfit2, fitness2, bestsol2, time2] = AOA(initsol, fname, xmin, xmax, Max_iter)

        print("DPO...")
        [bestfit3, fitness3, bestsol3, time3] = DPO(initsol, fname, xmin, xmax, Max_iter)

        print("CO...")
        [bestfit4, fitness4, bestsol4, time4] = CO(initsol, fname, xmin, xmax, Max_iter)

        print("Proposed")
        [bestfit5, fitness5, bestsol5, time5] = Proposed(initsol, fname, xmin, xmax, Max_iter)
        Bestsol.append([bestsol1, bestsol2, bestsol3, bestsol4, bestsol5])
    np.save('Bestsol_Feat.npy', Bestsol)

# optimized Weighted Fused Features
an = 0
if an:
    pca = PCA(n_components=1)
    Bestsol = np.load('Bestsol_Feat.npy', allow_pickle=True)
    for n in range(noOfDataset):
        Data = np.load('Data_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        learnperc = round(Data.shape[0] * 0.75)
        trainData = Data[:learnperc, :]
        testData = Data[learnperc:, :]
        trainTarget = Target[:learnperc, :]
        testTarget = Target[learnperc:, :]

        Feats, pred = Model_1DCNN(trainData, trainTarget, testData, testTarget)

        soln = Bestsol[n, 4, :]
        feat1 = Data[:, soln[:10].astype(np.int)] * soln[10:20]
        feat2 = Data[:, soln[20:30].astype(np.int)] * soln[30:40]
        feat3 = []
        for i in range(Data.shape[0]):
            print(n, i)
            data = Data[i, :]
            X_train = pca.fit_transform(data.reshape(-1, 1))
            X_test = pca.transform(data.reshape(-1, 1)).reshape(-1)[:10]
            feat3.append(X_test)
        feat3 = np.asarray(feat3)
        feat = soln[40] * feat1 + (1 - soln[40]) * feat2
        feat = soln[41] * feat + (1 - soln[41]) * feat3
        np.save('Feature' + str(n + 1) + '.npy', feat)

# Optimization for Classification
an = 0
if an:
    Bestsol = []
    for i in range(noOfDataset):
        Data = np.load('Feature' + str(i + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(i + 1) + '.npy', allow_pickle=True)

        learnperc = round(Data.shape[0] * 0.75)
        Global_vars.trainData = Data[:learnperc, :]
        Global_vars.testData = Data[learnperc:, :]
        Global_vars.trainTarget = Target[:learnperc, :]
        Global_vars.testTarget = Target[learnperc:, :]

        Npop = 10
        Chlen = 4
        xmin = matlib.repmat(np.asarray([2, 5, 2, 5]), Npop, 1)
        xmax = matlib.repmat(np.asarray([20, 255, 20, 255]), Npop, 1)
        initsol = np.zeros(xmax.shape)
        for p1 in range(Npop):
            for p2 in range(xmax.shape[1]):
                initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
        fname = objfun_LSTM
        Max_iter = 25

        print("HHO...")
        [bestfit1, fitness1, bestsol1, time1] = HHO(initsol, fname, xmin, xmax, Max_iter)

        print("AOA...")
        [bestfit2, fitness2, bestsol2, time2] = AOA(initsol, fname, xmin, xmax, Max_iter)

        print("DPO...")
        [bestfit3, fitness3, bestsol3, time3] = DPO(initsol, fname, xmin, xmax, Max_iter)

        print("CO...")
        [bestfit4, fitness4, bestsol4, time4] = CO(initsol, fname, xmin, xmax, Max_iter)

        print("Proposed")
        [bestfit5, fitness5, bestsol5, time5] = Proposed(initsol, fname, xmin, xmax, Max_iter)
        Bestsol.append([bestsol1, bestsol2, bestsol3, bestsol4, bestsol5])
    np.save('Bestsol_Cls.npy', Bestsol)

an = 0
if an:
    Bestsol = np.load('Bestsol_Cls.npy', allow_pickle=True)
    Learnper = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    Eval_all = []
    for n in range(noOfDataset):
        Data = np.load('Feature' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        Evall = []
        for i in range(len(Learnper)):
            Eval = np.zeros((10, 14))
            for j in range(Bestsol.shape[0]):
                sol = np.round(Bestsol[j, :]).astype(np.uint8)
                learnperc = round(Data.shape[0] * Learnper[i])
                Train_Data = Data[:learnperc, :]
                Train_Target = Target[:learnperc, :]
                Test_Data = Data[learnperc:, :]
                Test_Target = Target[learnperc:, :]
                Eval[j, :] = Model_ALSTM_DTCN(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Eval[5, :] = Model_LSTM(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[6, :], pred = Model_AttentionLSTM(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[7, :], pred = Model_DTCN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[8, :] = Model_ALSTM_DTCN(Train_Data, Train_Target, Test_Data, Test_Target)
            Eval[9, :] = Model_ALSTM_DTCN(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Evall.append(Eval)
        Eval_all.append(Evall)
    np.save('Old/Eval_all.npy', Eval_all)

plot_results()
plotresults()
plotConvResults()
