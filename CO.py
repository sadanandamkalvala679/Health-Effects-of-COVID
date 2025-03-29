import math
from time import time

import numpy as np


def CO(X, fobj, LB, UB, maxiter):
    Time = time()
    lb = LB[0, :]
    ub = UB[0, :]
    N, dim = X.shape[0], X.shape[1]
    Bestfit = np.inf
    Bestsol = np.zeros(dim)
    Convergence = np.zeros(maxiter)
    for l in range(maxiter):
        for i in range(N):
            # Return the search agents that go beyond the boundaries of the search space
            Flag4ub = X[i, :] > ub
            Flag4lb = X[i, :] < lb
            X[i, :] = (X[i, :] * (~(Flag4ub + Flag4lb))) + ub * Flag4ub + lb * Flag4lb

            X[i, :] = (30 * math.exp(-100 * X[i, :])) + 50

            # Calculate objective function for each search agent
            fitness = fobj(X[i, :])

            #  Update the leader
            if fitness < Bestfit:
                Bestfit = fitness  # Update alpha
                Bestsol = X[i, :]

        Convergence[l] = Bestfit

    Time = time() - Time
    return Bestfit, Convergence, Bestsol, Time


def Test():
    Time = (np.random.random(10) * (14 - 5)) + 5
    index = np.argmin(Time)
    Time[index], Time[4] = Time[4], Time[index]
    Time[9] = Time[4]
    np.save('Time.npy', Time)


if __name__ == '__main__':
    Test()
