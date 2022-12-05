import numpy as np
import pandas as pd
import time
import random
import math
from distance import Distance_EUC_2D
from dataProcess import read1PDPTW
from solnCheck import check1PDPTW, check1PDPTW_all
from utils import computeCost, getDistanceMatrix, get_x_from_soln
from collections import defaultdict

from docplex.mp.model import Model
from docplex.util.environment import get_environment


def solve1PDPTW_MIP_CPLEX(instance, timeLimit=600, verbose=0):
    startTime = time.time()
    timeSpent = time.time() - startTime

    # prep data
    M = 999999
    distMatrix = getDistanceMatrix(instance)
    V = range(instance['numLocation'] + 1) # set of vertices, create extra vertex for returning to depot
    P = [loc - 1 for loc in instance['pickup'] if loc != 0]  # set of pickup locations
    D = [loc - 1 for loc in instance['delivery'] if loc != 0] # set of delivery locations

    instance['demand'].append(0) # add 0 demand for artificial ending depot
    instance['tw'].append(instance['tw'][0]) # add tw for artificial ending depot

    mdl = Model('1PDPTW_cplex')
    mdl.parameters.timelimit(timeLimit)

    # Decision variables
    x = mdl.binary_var_matrix(keys1 = V, keys2 = V, name = 'x') # x_ij = 1 is location j is visited after location i
    s = mdl.integer_var_list(keys=V, lb=0, name = 's')
    q = mdl.integer_var_list(keys=V, lb=0, name = 'q')

    # objective function: minimize the distance
    total_distance = mdl.sum(distMatrix[i][j] * x[i,j] for i in V for j in V)

    # constraints

    # each location is only visited once
    for i in (P + D + [0]): # out
        mdl.add_constraint(mdl.sum(x[i,j] for j in (P + D + [len(V) - 1]) if i != j) == 1, 'out_once')
    for j in (P + D + [len(V) - 1]): # in
        mdl.add_constraint(mdl.sum(x[i,j] for i in (P + D + [0]) if i != j) == 1, 'in_once')

    # define s and q
    for i in V:
        for j in V:
            mdl.add_constraint(s[j] >= s[i] + distMatrix[i][j] - M * ( 1- x[i,j]))
            mdl.add_constraint(s[i] + distMatrix[i][j] >= s[j] -  M * ( 1 - x[i,j]))
            mdl.add_constraint(q[j] >= q[i] + instance['demand'][i] - M * ( 1- x[i,j]))
            mdl.add_constraint(q[i] + instance['demand'][i] >= q[j] - M * ( 1- x[i,j]))
    mdl.add_constraint(s[0] == 0) # make sure start time for depot is set to be 0
    mdl.add_constraint(q[0] == 0) # make sure load of vehicle is 0 at depot

    # tw
    for i in V:
        mdl.add_constraint(instance['tw'][i][0] <= s[i])
        mdl.add_constraint(s[i] <= instance['tw'][i][1])

    # capacity
    for i in V:
        mdl.add_constraint(q[i] <= instance['capacity'])

    # precedence
    for i in P:
        mdl.add_constraint(s[instance['delivery'][i] - 1] >= s[i] + distMatrix[i][instance['delivery'][i] - 1])

    # optimize
    mdl.minimize(total_distance)

    
    if mdl.solve():
        # get results
        soln = [0 + 1]
        curLoc = 0
        route = f'{0 + 1}'
        s_soln = [s[curLoc]]
        tt = []
        for i in range(len(V) - 1):
            # print([int(x[curLoc,j].x) for j in V])
            nextLoc = [int(round(x[curLoc,j].solution_value)) for j in V].index(1)
            route += (f' -> {nextLoc + 1}')
            soln.append(nextLoc + 1)
            s_soln.append(s[nextLoc])
            tt.append(distMatrix[curLoc][nextLoc])
            curLoc = nextLoc

        cost = computeCost(soln[0:-1], instance)
    
    else:
        soln = [0]
        cost = 999999

    return soln[0:-1], cost, timeSpent, mdl.get_solve_status()

if __name__ == "__main__":
    instance = read1PDPTW('data/1PDPTW_generated_d21_i1000_tmin300_tmax500_sd2022_test/INSTANCES/generated-301.txt')
    # instance = read1PDPTW('data/1PDPTW_generated/INSTANCES/generated-11-0.txt')
    # print(getDistanceMatrix(instance))
    soln, cost, solve_time, status = solve1PDPTW_MIP_CPLEX(instance, timeLimit = 10)
    print(soln, cost, solve_time, status)
