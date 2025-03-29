import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable


def plot_results():
    eval = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['MEP', 'SMAPE', 'MASE', 'MAE', 'RMSE', 'L1-NORM', 'L2-NORM', 'L1-INFITY NORM']
    Graph_Term = [0, 2, 3, 4]  # np.arange(len(Terms))  # [0, 3, 4, 5, 9]
    Algorithm = ['TERMS', 'HHO', 'AVOA', 'DPO', 'CO', 'PROPOSED']
    Classifier = ['TERMS', 'LSTM', 'ATTENTION LSTM', 'DTCN', 'ALSTM+DTCN', 'PROPOSED']

    for n in range(eval.shape[0]):
        value = eval[n, 4, :, :]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[j, :])
        print('---------------------------------------- Dataset ' + str(n + 1) + ' Algorithm Comparison',
              '----------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, :])
        print('---------------------------------------- Dataset ' + str(n + 1) + ' Classifier Comparison',
              '----------------------------------------')
        print(Table)

    X = np.arange(6)
    for n in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[1], eval.shape[2]))
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval[n, k, l, Graph_Term[j]]
                    else:
                        Graph[k, l] = eval[n, k, l, Graph_Term[j]] * 100

            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
            ax.plot(X, Graph[:, 0], color='r', linewidth=3, marker='o', markerfacecolor='blue',
                    markersize=12, label="HHO-DTCN-ALSTM")
            ax.plot(X, Graph[:, 1], color='g', linewidth=3, marker='o', markerfacecolor='red',
                    markersize=12, label="AVOA-DTCN-ALSTM")
            ax.plot(X, Graph[:, 2], color='b', linewidth=3, marker='o', markerfacecolor='green',
                    markersize=12, label="DPO-DTCN-ALSTM")
            ax.plot(X, Graph[:, 3], color='m', linewidth=3, marker='o', markerfacecolor='yellow',
                    markersize=12, label="CO-DTCN-ALSTM")
            ax.plot(X, Graph[:, 4], color='k', linewidth=3, marker='o', markerfacecolor='cyan',
                    markersize=12, label="HDPCO-DTCN-ALSTM")
            plt.xticks(X, ('8', '16', '24', '32', '48', '64'))  # , rotation=45
            plt.xlabel('Batch size')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc='best')
            path1 = "./Results/%s_%s_alg_1.png" % (n + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
            ax.bar(X + 0.00, Graph[:, 5], color='r', width=0.10, label="LSTM")
            ax.bar(X + 0.10, Graph[:, 6], color='g', width=0.10, label="ALSTM")
            ax.bar(X + 0.20, Graph[:, 7], color='b', width=0.10, label="DTCN")
            ax.bar(X + 0.30, Graph[:, 8], color='m', width=0.10, label="ALSTM-DTCN")
            ax.bar(X + 0.40, Graph[:, 9], color='k', width=0.10, label="HDPCO-DTCN-ALSTM")
            plt.xticks(X, ('8', '16', '24', '32', '48', '64'))  # , rotation=45
            plt.xlabel('Batch size')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc='best')
            path1 = "./Results/%s_%s_mtd_1.png" % (n + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()


def plotresults():
    eval = np.load('Eval_fold.npy', allow_pickle=True)
    Terms = ['MEP', 'SMAPE', 'MASE', 'MAE', 'RMSE', 'L1-NORM', 'L2-NORM', 'L1-INFITY NORM']
    Graph_Term = [0, 2, 3, 4]  # np.arange(len(Terms))  # [0, 3, 4, 5, 9]
    Algorithm = ['TERMS', 'HHO', 'AVOA', 'DPO', 'CO', 'PROPOSED']
    Classifier = ['TERMS', 'LSTM', 'ATTENTION LSTM', 'DTCN', 'ALSTM+DTCN', 'PROPOSED']

    # for n in range(eval.shape[0]):
    #     value = eval[n, 4, :, :]
    #
    #     Table = PrettyTable()
    #     Table.add_column(Algorithm[0], Terms)
    #     for j in range(len(Algorithm) - 1):
    #         Table.add_column(Algorithm[j + 1], value[j, :])
    #     print('---------------------------------------- Dataset ' + str(n + 1) + ' Algorithm Comparison',
    #           '----------------------------------------')
    #     print(Table)
    #
    #     Table = PrettyTable()
    #     Table.add_column(Classifier[0], Terms)
    #     for j in range(len(Classifier) - 1):
    #         Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, :])
    #     print('---------------------------------------- Dataset ' + str(n + 1) + ' Classifier Comparison',
    #           '----------------------------------------')
    #     print(Table)

    X = np.arange(5)
    for n in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[1], eval.shape[2]))
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval[n, k, l, Graph_Term[j]]
                    else:
                        Graph[k, l] = eval[n, k, l, Graph_Term[j]] * 100

            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
            ax.plot(X, Graph[:, 0], color='r', linewidth=3, marker='o', markerfacecolor='blue',
                    markersize=12, label="HHO-DTCN-ALSTM")
            ax.plot(X, Graph[:, 1], color='g', linewidth=3, marker='o', markerfacecolor='red',
                    markersize=12, label="AVOA-DTCN-ALSTM")
            ax.plot(X, Graph[:, 2], color='b', linewidth=3, marker='o', markerfacecolor='green',
                    markersize=12, label="DPO-DTCN-ALSTM")
            ax.plot(X, Graph[:, 3], color='m', linewidth=3, marker='o', markerfacecolor='yellow',
                    markersize=12, label="CO-DTCN-ALSTM")
            ax.plot(X, Graph[:, 4], color='k', linewidth=3, marker='o', markerfacecolor='cyan',
                    markersize=12, label="HDPCO-DTCN-ALSTM")
            plt.xticks(X, ('RELU', 'SIGMOID', 'SOFMAX', 'LINEAR', 'TANH'))  # , rotation=45
            plt.xlabel('Activation Functions')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc='best')
            path1 = "./Results/%s_%s_alg_2.png" % (n + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
            ax.bar(X + 0.00, Graph[:, 5], color='r', width=0.10, label="LSTM")
            ax.bar(X + 0.10, Graph[:, 6], color='g', width=0.10, label="ALSTM")
            ax.bar(X + 0.20, Graph[:, 7], color='b', width=0.10, label="DTCN")
            ax.bar(X + 0.30, Graph[:, 8], color='m', width=0.10, label="ALSTM-DTCN")
            ax.bar(X + 0.40, Graph[:, 9], color='k', width=0.10, label="HDPCO-DTCN-ALSTM")
            plt.xticks(X, ('RELU', 'SIGMOID', 'SOFMAX', 'LINEAR', 'TANH'))  # , rotation=45
            plt.xlabel('Activation Functions')
            plt.ylabel(Terms[Graph_Term[j]])
            plt.legend(loc='best')
            path1 = "./Results/%s_%s_mtd_2.png" % (n + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'HHO', 'AVOA', 'DPO', 'CO', 'PROPOSED']

    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    for n in range(Fitness.shape[0]):
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[n, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Report ',
              '--------------------------------------------------')
        print(Table)

        length = np.arange(Fitness.shape[2])
        Conv_Graph = Fitness[n]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label='HHO-DTCN-ALSTM')
        plt.plot(length, Conv_Graph[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label='AVOA-DTCN-ALSTM')
        plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label='DPO-DTCN-ALSTM')
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label='CO-DTCN-ALSTM')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label='HDPCO-DTCN-ALSTM')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/Conv" + str(n + 1) + ".png")
        plt.show()


def PlotTime():
    # matplotlib.use('TkAgg')
    Time = np.load('Time.npy', allow_pickle=True)
    Algorithm = ['Optimization Algorithms', 'HHO', 'AVOA', 'DPO', 'CO', 'PROPOSED']
    Classifier = ['Networks', 'LSTM', 'ATTENTION LSTM', 'DTCN', 'ALSTM+DTCN', 'PROPOSED']

    Table1 = PrettyTable()
    Table1.add_column(Algorithm[0], Algorithm[1:])
    Table1.add_column('Time Complexity', Time[:5])
    print('------------------------------ Time Complexity for Algorithms ------------------------------')
    print(Table1)

    Table2 = PrettyTable()
    Table2.add_column(Classifier[0], Classifier[1:])
    Table2.add_column('Time Complexity', Time[5:])
    print('------------------------------ Time Complexity for Classifiers ------------------------------')
    print(Table2)


if __name__ == '__main__':
    # plot_results()
    # plotresults()
    # plotConvResults()
    PlotTime()
