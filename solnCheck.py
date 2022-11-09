# Check if a solution is feasible

from distance import Distance_EUC_2D
from collections import Counter
from dataProcess import read1PDPTW

def check1PDPTW(soln, instance, return_now=True):
    precedence_check = True
    tw_check = True
    capacity_check = True
    error = []

    curTime = 0
    curLoad = 0
    curLoc = 0
    visited_loc = [1]

    if len(soln) != instance['numLocation']:
        error.append('LOCATION ERROR: not all locations have been visited')
    else:
        for loc in Counter(soln):
            if Counter(soln)[loc] > 1:
                error.append(f'LOCATION ERROR: location {loc} is visited {Counter(soln)[loc]} times')
    
    capacity = instance['capacity']

    while (curLoc < len(soln) - 1):
        curLoc += 1
        # print(curLoc)
        next_loc = soln[curLoc]
        # print(next_loc)
        visited_loc.append(next_loc)
        curTime += Distance_EUC_2D(instance['coordinates'][soln[curLoc-1]-1], instance['coordinates'][soln[curLoc]-1])
        est = instance['tw'][soln[curLoc]-1][0]
        lft = instance['tw'][soln[curLoc]-1][1]

        # print(curTime, ' ; ', est, ' ; ', lft)
        if (curTime < est) or (curTime > lft):
            tw_check = False
            error.append(f'TW ERROR: location {next_loc}, tw = ({est}, {lft}), arrival time = {curTime}')
        
        curLoad += instance['demand'][soln[curLoc]-1]
        if curLoad > capacity:
            capacity_check = False
            error.append(f'CAPACITY ERROR: capacity = {capacity}, load = {curLoad}')

        delivery_loc = instance['delivery'][soln[curLoc]-1]
        pickup_loc = instance['pickup'][soln[curLoc]-1]
        if delivery_loc == 0: # if location visited is a delivery location
            if  pickup_loc not in visited_loc: # if associated pickup location has not been visited prior
                precedence_check = False
                error.append(f'PRECEDENCE ERROR: location {next_loc} visited before location {pickup_loc}')

        if return_now:
            if not precedence_check or not tw_check or not capacity_check:
                break

    return precedence_check, tw_check, capacity_check, error, curLoc, curTime

# instance = read1PDPTW('test_data/generated-1113.txt')
instance = read1PDPTW('data/1PDPTW_generated/INSTANCES/generated-11-0.txt')
soln1039 = [1, 5, 9, 4, 8, 6, 7, 10, 2, 11, 3]
soln11_0_opt = [1, 3, 11, 2, 5, 7, 8, 6, 4, 9, 10]
soln2 = [1,3,7,11,9,2,4,5,6,10,8]
# soln1 = [1, 10, 7, 11, 9, 2, 8, 3, 4]
# soln2 = [1, 10, 7, 5, 9, 2, 5, 8, 6, 3, 4]
# soln3 = [1, 5, 3, 7, 9, 10, 2, 4, 6, 8, 11]
precedence_check, tw_check, capacity_check, error, violatedLoc, curTime = check1PDPTW(soln11_0_opt, instance)

print(error)