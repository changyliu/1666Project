import numpy as np
import time
import random
import math
from distance import Distance_EUC_2D
from dataProcess import read1PDPTW
from solnCheck import check1PDPTW
from utils import computeCost

def generateRandomSoln(dimension, solnHistory):
    seen = True
    while seen:
        newSoln = [1]
        newSoln += random.sample(range(2, instance['numLocation'] + 1), instance['numLocation'] - 1)
        if newSoln not in solnHistory:
            seen = False
    return newSoln

def solnRepair(soln, instance, iterLimit, timeLimit, verbose=0):
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
            # print(violatedLoc)
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

            print(newSoln)
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
                newSoln = generateRandomSoln(instance['numLocations'], solnHistory)

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


if __name__ == "__main__":
    # soln1 = [1,11,7,9,10,6,5,2,8,3,4]
    # soln2 = [1,10,9,11,5,2,8,3,4,7,6]
    # soln3 = [1,10,7,11,9,5,6,2,8,3,4]

    soln1 = [1, 5, 9, 4, 8, 6, 7, 10, 2, 11, 3]
    soln1039 = [1, 5, 9, 6, 10, 7, 4, 8, 2, 3, 11]
    instance = read1PDPTW('test_data/generated-1039.txt')
    
    # print(instance['tw'])

    repairedSoln, numIter, timeSpent, solnFeasibility = solnRepair(soln1039, instance, 200, 600,1)
    cost = 999999
    if solnFeasibility:
        cost = computeCost(repairedSoln, instance)
    print('\n')
    print(repairedSoln)
    print(f'Cost: {cost}')
    print(f'Total iterations: {numIter}, Time Spent: {timeSpent}')
