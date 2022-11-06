import numpy as np
import random

def generateFeasiblePDTour(dimension, pickup, delivery, randSeed):
    random.seed(randSeed)

    availablePool = pickup.copy()
    soln = []
    soln.append(1) # always start at depot
    for i in range(dimension - 1):
        # print(i)
        # print(availablePool)
        nextLoc = random.sample(availablePool, 1)[0]
        # print(nextLoc)
        soln.append(nextLoc)
        availablePool.remove(nextLoc) # remove chosen location
        if nextLoc in pickup:
            availablePool.append(delivery[pickup.index(nextLoc)]) # if pickup location, add its respective delivery location

    return soln
