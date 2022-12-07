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

def generateRandomSoln(dimension, solnHistory, rand_seed):
    random.seed(rand_seed)
    seen = True
    while seen:
        newSoln = [1]
        newSoln += random.sample(range(2, dimension + 1), dimension - 1)
        if newSoln not in solnHistory:
            seen = False
    return newSoln

def localSearch(soln, instance, iterLimit, timeLimit, verbose=0):
    startTime = time.time()
    timeSpent = time.time() - startTime
    i = 0

    precedence_check, tw_check, capacity_check, error, violatedLoc, locTime = check1PDPTW(soln, instance)
    newSoln = soln.copy()
    solnHistory = [soln]

    while (not precedence_check or not tw_check or not capacity_check) and i < iterLimit and timeSpent < timeLimit and len(solnHistory) < math.factorial(instance['numLocation']-1):
        if verbose:
            print(f'***** ITERATION {i} *****')
            print(error)
            print(newSoln)
            print(violatedLoc)
        if not precedence_check: # insert associated pickup location before current delivery location
            newSoln.remove(instance['pickup'][soln[violatedLoc]-1])
            newSoln.insert(violatedLoc, instance['pickup'][soln[violatedLoc]-1])

            # check if newSoln has been tried previously:
            if newSoln in solnHistory:
                newSoln = generateRandomSoln(instance['numLocation'], solnHistory)

            precedence_check, tw_check, capacity_check, error, violatedLoc, locTime = check1PDPTW(newSoln, instance)
            soln = newSoln.copy()
            solnHistory.append(soln)

        elif not tw_check: # swap with neighbouring location
            # if vehicle arrives too early
            if locTime < instance['tw'][soln[violatedLoc] - 1][0]:
                # check if cur location is pickup location and if next location is its delivery loc
                if (instance['pickup'][soln[violatedLoc] - 1] == 0) and (soln[violatedLoc + 1] == instance['delivery'][soln[violatedLoc]-1]):
                    # check if these are the last pair
                    if violatedLoc + 2 > instance['numLocation'] - 1:
                        # restart from random solution
                        generateRandomSoln(instance['numLocation'], solnHistory)
                    else:
                        newSoln.remove(soln[violatedLoc + 2]) # move both locations to later
                        newSoln.insert(violatedLoc, soln[violatedLoc + 2])
                else: # swap with next location
                    # check if this is the last location
                    if violatedLoc + 1 > instance['numLocation'] - 1:
                        # restart from random solution
                        generateRandomSoln(instance['numLocation'], solnHistory)
                    else:
                        newSoln.remove(soln[violatedLoc + 1])
                        newSoln.insert(violatedLoc, soln[violatedLoc + 1])
            # if vehicle arrives too late
            elif locTime > instance['tw'][soln[violatedLoc] - 1][1]:
                # check if cur location is delivery location and if previous location is its pickup location
                if (instance['delivery'][soln[violatedLoc]-1] == 0) and (soln[violatedLoc - 1] == instance['pickup'][soln[violatedLoc] - 1]):
                    # check if this is the first pair, then no feasible solution exists
                    if violatedLoc - 2 < 1:
                        print('No feasible solution exists')
                        return [], i, timeSpent, (precedence_check and tw_check and capacity_check)
                    else:
                        newSoln.remove(soln[violatedLoc - 2]) # then move both locations ahead
                        newSoln.insert(violatedLoc, soln[violatedLoc - 2])
                else: # else swap with previous location
                    # check if this is the first location, then no feasible solution exists
                    if violatedLoc - 1 < 1:
                        print('No feasible solution exists')
                        return [], i, timeSpent, (precedence_check and tw_check and capacity_check)
                    else:
                        newSoln.remove(soln[violatedLoc - 1])
                        newSoln.insert(violatedLoc, soln[violatedLoc - 1])

            # check if newSoln has been tried previously:
            if newSoln in solnHistory:
                newSoln = generateRandomSoln(instance['numLocation'], solnHistory)

            precedence_check, tw_check, capacity_check, error, violatedLoc, locTime = check1PDPTW(newSoln, instance)
            soln = newSoln.copy()
            solnHistory.append(soln)

        elif not capacity_check: # visit delivery location of latest undropped off pickup before current node (offload some weight)
            # find latest pickup location visited
            pickupLoc = []
            # print(soln[1:violatedLoc])
            for loc in soln[1:violatedLoc]:
                if instance['pickup'][loc - 1] == 0: # check if pickup location
                    pickupLoc.append(loc)
                else: # if it's a delivery location, then we remove its associated pickup location from pickupLoc, assuming we have no precedence violations until now
                    pickupLoc.remove(instance['pickup'][loc - 1])
                # print(pickupLoc)

            latestPickup = pickupLoc[-1]
            newSoln.remove(instance['delivery'][latestPickup - 1])
            newSoln.insert(violatedLoc, instance['delivery'][latestPickup - 1])

            # check if newSoln has been tried previously:
            if newSoln in solnHistory:
                newSoln = generateRandomSoln(instance['numLocation'], solnHistory)

            precedence_check, tw_check, capacity_check, error, violatedLoc, locTime = check1PDPTW(newSoln, instance)
            soln = newSoln.copy()
            solnHistory.append(soln)

        i += 1
        timeSpent = time.time() - startTime

        if i >= iterLimit:
            print('Iteration limit reached and no feasible solution has been found')
            return [], i, timeSpent, (precedence_check and tw_check and capacity_check)

        if timeSpent >= timeLimit:
            print('Time limit reached and no feasible solution has been found')
            return [], i, timeSpent, (precedence_check and tw_check and capacity_check)

    return newSoln, i, timeSpent, (precedence_check and tw_check and capacity_check)

def localSearchExtended(soln, instance, iterLimit, timeLimit, strategy = 0, verbose=0):
    # stategies: 0 -> fisrt encountered violation, 1 -> most violated location, 2 -> least violated location

    startTime = time.time()
    timeSpent = time.time() - startTime
    i = 0
    num_dict = defaultdict(int)
    # num_restart = 0
    # num_tw_violation = 0
    # num_precedence_violation = 0
    # num_capacity_violation = 0

    precedence_check, tw_check, capacity_check, error, error_dict, violatedLoc, locTime = check1PDPTW_all(soln, instance)
    newSoln = soln.copy()
    solnHistory = [soln]

    while (not precedence_check or not tw_check or not capacity_check) and i < iterLimit and timeSpent < timeLimit and len(solnHistory) < math.factorial(instance['numLocation']-1):
        violatedLoc, error = get_violated_loc(error_dict, strategy)

        if verbose:
            print(f'***** ITERATION {i} *****')
            print(pd.DataFrame(error_dict))
            print(error)
            print(newSoln)
            print(violatedLoc)

        if error['type'] == 'PRECEDENCE': # insert associated pickup location before current delivery location
            newSoln.remove(instance['pickup'][soln[violatedLoc]-1])
            newSoln.insert(violatedLoc, instance['pickup'][soln[violatedLoc]-1])

            # check if newSoln has been tried previously:
            if newSoln in solnHistory:
                num_dict['restart'] += 1
                newSoln = generateRandomSoln(instance['numLocation'], solnHistory, num_dict['restart'])

                # if verbose:
                #     print('Random restart')

            num_dict['precedence_violation'] += 1

        elif error['type'] == 'TW': # swap with neighbouring location
            # if vehicle arrives too early
            # print(error['range'])
            # print(instance['tw'][soln[violatedLoc] - 1])
            if error['current'] < instance['tw'][soln[violatedLoc] - 1][0]:
                # check if cur location is pickup location and if next location is its delivery loc
                if (instance['pickup'][soln[violatedLoc] - 1] == 0) and (soln[violatedLoc + 1] == instance['delivery'][soln[violatedLoc]-1]):
                    # check if these are the last pair
                    if violatedLoc + 2 > instance['numLocation'] - 1:
                        # restart from random solution
                        generateRandomSoln(instance['numLocation'], solnHistory, num_dict['restart'])
                        num_dict['restart'] += 1
                    else:
                        newSoln.remove(soln[violatedLoc + 2]) # move both locations to later
                        newSoln.insert(violatedLoc, soln[violatedLoc + 2])

                        if verbose:
                            print('tw - move both locations to later')
                else: # swap with next location
                    # check if this is the last location
                    if violatedLoc + 1 > instance['numLocation'] - 1:
                        # restart from random solution
                        generateRandomSoln(instance['numLocation'], solnHistory, num_dict['restart'])
                        num_dict['restart'] += 1
                    else:
                        newSoln.remove(soln[violatedLoc + 1])
                        newSoln.insert(violatedLoc, soln[violatedLoc + 1])
                        if verbose:
                            print('tw - move location to later')
            # if vehicle arrives too late
            elif error['current'] > instance['tw'][soln[violatedLoc] - 1][1]:
                # check if cur location is delivery location and if previous location is its pickup location
                if (instance['delivery'][soln[violatedLoc]-1] == 0) and (soln[violatedLoc - 1] == instance['pickup'][soln[violatedLoc] - 1]):
                    # check if this is the first pair, then no feasible solution exists
                    if violatedLoc - 2 < 1:
                        print('No feasible solution exists')
                        return [], i, timeSpent, (precedence_check and tw_check and capacity_check)
                    else:
                        newSoln.remove(soln[violatedLoc - 2]) # then move both locations ahead
                        newSoln.insert(violatedLoc, soln[violatedLoc - 2])
                        if verbose:
                            print('tw - move both locations ahead')
                else: # else swap with previous location
                    # check if this is the first location, then no feasible solution exists
                    if violatedLoc - 1 < 1:
                        print('No feasible solution exists')
                        return [], i, timeSpent, (precedence_check and tw_check and capacity_check)
                    else:
                        newSoln.remove(soln[violatedLoc - 1])
                        newSoln.insert(violatedLoc, soln[violatedLoc - 1])
                        if verbose:
                            print('tw - move location ahead')

            # check if newSoln has been tried previously:
            if newSoln in solnHistory:
                newSoln = generateRandomSoln(instance['numLocation'], solnHistory, num_dict['restart'])
                num_dict['restart'] += 1

            num_dict['tw_violation'] += 1

        elif error['type'] == 'CAPACITY': # visit delivery location of latest undropped off pickup before current node (offload some weight)
            # find latest pickup location visited
            pickupLoc = []
            for loc in soln[1:violatedLoc]:
                if instance['pickup'][loc - 1] == 0: # check if pickup location
                    pickupLoc.append(loc)
                else: # if it's a delivery location, then we remove its associated pickup location from pickupLoc, assuming we have no precedence violations until now
                    if instance['pickup'][loc - 1] in pickupLoc:
                        pickupLoc.remove(instance['pickup'][loc - 1])

            latestPickup = pickupLoc[-1]
            newSoln.remove(instance['delivery'][latestPickup - 1])
            newSoln.insert(violatedLoc, instance['delivery'][latestPickup - 1])

            # check if newSoln has been tried previously:
            if newSoln in solnHistory:
                newSoln = generateRandomSoln(instance['numLocation'], solnHistory, num_dict['restart'])
                num_dict['restart'] += 1

            num_dict['capacity_violation'] += 1


        precedence_check, tw_check, capacity_check, error, error_dict, violatedLoc, locTime = check1PDPTW_all(newSoln, instance)
        soln = newSoln.copy()
        solnHistory.append(soln)

        i += 1
        timeSpent = time.time() - startTime

        if i >= iterLimit:
            print('Iteration limit reached and no feasible solution has been found')
            return [], i, timeSpent, num_dict, (precedence_check and tw_check and capacity_check)

        if timeSpent >= timeLimit:
            print('Time limit reached and no feasible solution has been found')
            return [], i, timeSpent, num_dict, (precedence_check and tw_check and capacity_check)

    return newSoln, i, timeSpent, num_dict, (precedence_check and tw_check and capacity_check)

def get_violated_loc(error_dict, strategy):
    error_df = pd.DataFrame(error_dict)

    if strategy == 0:
        violatedLoc = error_df['loc'][0]
        error = error_df.loc[0]
    elif strategy == 1:
        violatedLoc = error_df['loc'][error_df['violation'].idxmax()]
        error = error_df.loc[error_df['violation'].idxmax()]
    elif strategy == 2:
        violatedLoc = error_df['loc'][error_df['violation'].idxmin()]
        error = error_df.loc[error_df['violation'].idxmin()]

    return violatedLoc, error

def cplex_MIP(rl_soln, instance, iterLimit, timeLimit, verbose=0):
    # TODO: add time and iteration limits

    x_rl_soln = get_x_from_soln(rl_soln)

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

    mdl = Model('1PDPTW_cplex_repair')

    # Decision variables
    x = mdl.binary_var_matrix(keys1 = V, keys2 = V, name = 'x') # x_ij = 1 is location j is visited after location i
    s = mdl.integer_var_list(keys=V, lb=0, name = 's')
    q = mdl.integer_var_list(keys=V, lb=0, name = 'q')

    # objective function: minimize the distance
    # total_distance = mdl.sum(distMatrix[i][j] * x[i,j] for i in V for j in V)

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

    values = {x[i,j]: x_rl_soln[i][j] for i in V for j in V}

    warmstart=mdl.new_solution()
    warmstart.update( values )
    mdl.add_mip_start(warmstart, effort_level = 4) # level 4 is soln repair https://www.ibm.com/docs/en/icos/20.1.0?topic=mip-starting-from-solution-starts

    # optimize
    # mdl.minimize(total_distance)

    mdl.solve()


    # get results

    soln = [0 + 1]
    curLoc = 0
    route = f'{0 + 1}'
    s_soln = [s[curLoc]]
    tt = []
    for i in range(len(V) - 1):
        # print([int(x[curLoc,j].x) for j in V])
        nextLoc = [int(x[curLoc,j]) for j in V].index(1)
        route += (f' -> {nextLoc + 1}')
        soln.append(nextLoc + 1)
        s_soln.append(s[nextLoc])
        tt.append(distMatrix[curLoc][nextLoc])
        curLoc = nextLoc

    cost = computeCost(soln[0:-1], instance)

    # print('\n')
    # print(soln)
    # print(f'Route: {route}')
    # print(f'Cost: {cost}')
    # # print(tt)
    # print(s_soln)

    # timeSpent = time.time() - startTime

    # if timeSpent >= timeLimit:
    #     print('Time limit reached and no feasible solution has been found')
    #     # return [], i, timeSpent, (precedence_check and tw_check and capacity_check)

    return soln[0:-1], cost, timeSpent


if __name__ == "__main__":
    # soln1 = [1,11,7,9,10,6,5,2,8,3,4]
    # soln2 = [1,10,9,11,5,2,8,3,4,7,6]
    # soln3 = [1,10,7,11,9,5,6,2,8,3,4]

    soln1 = [1, 5, 9, 4, 8, 6, 7, 10, 2, 11, 3]
    soln1113 = [1, 8, 7, 6, 10, 3, 9, 11, 4, 2, 5]
    instance = read1PDPTW('data/1PDPTW_generated_d15_i1000_tmin300_tmax500_sd2022_test/INSTANCES/generated-345.txt')

    # print(instance['tw'])

    # repairedSoln, cost, timeSpent = cplex_MIP(soln1039, instance, 200, 600, verbose=0)
    # print('\n')
    # print(repairedSoln)
    # print(f'Cost: {cost}')
    # print(f'Time Spent: {timeSpent}')

    # repairedSoln, numIter, timeSpent, solnFeasibility = localSearch(soln1039, instance, 200, 600,0)
    # cost = 999999
    # if solnFeasibility:
    #     cost = computeCost(repairedSoln, instance)
    # print('\n')
    # print(repairedSoln)
    # print(f'Cost: {cost}')
    # print(f'Total iterations: {numIter}, Time Spent: {timeSpent}')

    soln1039 = [0, 8, 13, 7, 1, 5, 3, 12, 14, 4, 2, 10, 6, 11, 9]

    # print('####### first violation #######')
    # repairedSoln, numIter, timeSpent, num_dict, solnFeasibility = localSearchExtended(soln1039, instance, 500, 600, strategy = 0, verbose=0)

    # cost = 999999
    # if solnFeasibility:
    #     cost = computeCost(repairedSoln, instance)
    # print('\n')
    # print(repairedSoln)
    # print(f'Cost: {cost}')
    # print(f'Iterations: {numIter}')
    # print(f'Time Spent: {timeSpent}')
    # print('Number of restart: ', num_dict['restart'])

    # print('####### most violation #######')
    # repairedSoln, numIter, timeSpent, num_dict, solnFeasibility = localSearchExtended(soln1039, instance, 500, 600, strategy = 1, verbose=0)
    # cost = 999999
    # if solnFeasibility:
    #     cost = computeCost(repairedSoln, instance)
    # print('\n')
    # print(repairedSoln)
    # print(f'Cost: {cost}')
    # print(f'Iterations: {numIter}')
    # print(f'Time Spent: {timeSpent}')
    # print('Number of restart: ', num_dict['restart'])

    print('####### least violation #######')
    repairedSoln, numIter, timeSpent, num_dict, solnFeasibility = localSearchExtended(soln1039, instance, 500, 600, strategy = 2, verbose=0)
    cost = 999999
    if solnFeasibility:
        cost = computeCost(repairedSoln, instance)
    print('\n')
    print(repairedSoln)
    print(f'Cost: {cost}')
    print(f'Iterations: {numIter}')
    print(f'Time Spent: {timeSpent}')
    print('Number of restart: ', num_dict['restart'])
