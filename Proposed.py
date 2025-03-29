import math
from time import time

import numpy as np


def Proposed(X, fobj, LB, UB, maxiter):
    Time = time()
    lb = LB[0, :]
    ub = UB[0, :]
    N, dim = X.shape[0], X.shape[1]
    Bestfit = np.inf
    Bestsol = np.zeros(dim)
    fitness = fobj(X)
    Convergence = np.zeros(maxiter)
    for l in range(maxiter):
        for i in range(N):
            # Return the search agents that go beyond the boundaries of the search space
            Flag4ub = X[i, :] > ub
            Flag4lb = X[i, :] < lb
            X[i, :] = (X[i, :] * (~(Flag4ub + Flag4lb))) + ub * Flag4ub + lb * Flag4lb
            r = 2 * ((np.mean(fitness) - np.min(fitness)) / (np.max(fitness) - np.min(fitness))) + 2

            if r > 3:
                rand_index = math.floor(N * np.random.random())
                X_rand = X[rand_index, :]
                X[i, :] = X_rand - np.random.random() * abs(X_rand - 2 * np.random.random() * X[i, :])
            else:
                X[i, :] = (30 * math.exp(-100 * X[i, :])) + 50

            # Calculate objective function for each search agent
            fitness[i] = fobj(X[i, :])

            #  Update the leader
            if fitness[i] < Bestfit:
                Bestfit = fitness[i]  # Update alpha
                Bestsol = X[i, :]

        Convergence[l] = Bestfit

    Time = time() - Time
    return Bestfit, Convergence, Bestsol, Time
