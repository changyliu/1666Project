import numpy as np
import time
from distance import Distance_EUC_2D
from dataProcess import read1PDPTW
from solnCheck import check1PDPTW

def computeCost(soln, instance):
    totalTravelTime = 0
    for i in range(instance['numLocation']-1):
        totalTravelTime += Distance_EUC_2D(instance['coordinates'][soln[i] - 1], instance['coordinates'][soln[i+1] - 1])

    totalTravelTime += Distance_EUC_2D(instance['coordinates'][soln[-1] - 1], instance['coordinates'][soln[0]]) # add time to return to depot
    return totalTravelTime

def solnRepair(soln, instance, iterLimit, timeLimit, verbose=0):
    startTime = time.time()
    timeSpent = time.time() - startTime
    i = 0

    precedence_check, tw_check, capacity_check, error, violatedLoc, locTime = check1PDPTW(soln, instance)
    newSoln = soln.copy()
    solnHistory = [soln]

    while (not precedence_check or not tw_check or not capacity_check) and i < iterLimit and timeSpent < timeLimit:
        if verbose:
            print(f'***** ITERATION {i} *****')
            print(error)
            print(newSoln)
            # print(violatedLoc)
        if not precedence_check: # insert associated pickup location before current delivery location
            newSoln.remove(instance['pickup'][soln[violatedLoc]-1])
            newSoln.insert(violatedLoc, instance['pickup'][soln[violatedLoc]-1])
            precedence_check, tw_check, capacity_check, error, violatedLoc, locTime = check1PDPTW(newSoln, instance)
            soln = newSoln.copy()
            solnHistory.append(soln)

        elif not tw_check: # swap with neighbouring location
            # if vehicle arrives too early    
            if locTime < instance['tw'][soln[violatedLoc] - 1][0]:
                # check if cur location is pickup location and if next location is its delivery loc
                if (instance['pickup'][soln[violatedLoc] - 1] == 0) and (soln[violatedLoc + 1] == instance['delivery'][soln[violatedLoc]-1]):
                    newSoln.remove(soln[violatedLoc + 2]) # move both locations to later
                    newSoln.insert(violatedLoc, soln[violatedLoc + 2])
                else: # swap with next location
                    newSoln.remove(soln[violatedLoc + 1])
                    newSoln.insert(violatedLoc, soln[violatedLoc + 1])
            elif locTime > instance['tw'][soln[violatedLoc] - 1][1]:
                # check if cur location is delivery location and if previous location is its pickup location
                if (instance['delivery'][soln[violatedLoc]-1] == 0) and (soln[violatedLoc - 1] == instance['pickup'][soln[violatedLoc] - 1]):
                    newSoln.remove(soln[violatedLoc - 2]) # then move both locations ahead
                    newSoln.insert(violatedLoc, soln[violatedLoc - 2])
                else: # else swap with previous location
                    newSoln.remove(soln[violatedLoc - 1])
                    newSoln.insert(violatedLoc, soln[violatedLoc - 1])
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
            precedence_check, tw_check, capacity_check, error, violatedLoc, locTime = check1PDPTW(newSoln, instance)
            soln = newSoln.copy()
            solnHistory.append(soln)

        i += 1
        timeSpent = time.time() - startTime
    return newSoln, i, timeSpent, (precedence_check and tw_check and capacity_check)


if __name__ == "__main__":
    soln1 = [1,11,7,9,10,6,5,2,8,3,4]
    soln2 = [1,10,9,11,5,2,8,3,4,7,6]
    soln3 = [1,10,7,11,9,5,6,2,8,3,4]
    instance = read1PDPTW('data/1PDPTW_generated/INSTANCES/generated-0.txt')

    repairedSoln, numIter, timeSpent = solnRepair(soln2, instance, 5000, 600)
    cost = computeCost(repairedSoln, instance)
    print('\n')
    print(repairedSoln)
    print(f'Cost: {cost}')
    print(f'Total iterations: {numIter}, Time Spent: {timeSpent}')
