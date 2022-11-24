import numpy as np
import random
import os
from distance import Distance_EUC_2D
from solnGenerator import generateFeasiblePDTour
from dataProcess import read1PDPTW, read1PDPTW_tour
from solnCheck import check1PDPTW

def generate_1PDPTW(dimension, numInstance, randSeed, tw_min=100, tw_max=300, ext=''):
    os.makedirs("./data/1PDPTW_generated_d{}_i{}_tmin{}_tmax{}_sd{}{}/INSTANCES".format(dimension, numInstance, tw_min, tw_max, randSeed, ext), exist_ok=True)
    os.makedirs("./data/1PDPTW_generated_d{}_i{}_tmin{}_tmax{}_sd{}{}/TOURS".format(dimension, numInstance, tw_min, tw_max, randSeed, ext), exist_ok=True)

    vehicleNum = 1
    serviceTime = 0

    random.seed(randSeed)

    randomSeeds = [random.randint(0,5000) for i in range(numInstance)]

    for i in range(numInstance):

        random.seed(randomSeeds[i])

        lines = []

        name = f'generated-{i}'

        pickup = random.sample(range(2, dimension+1), int((dimension-1)/2)) # 1 is depot location, no pickup or delivery
        delivery = [x for x in range(2, dimension+1) if x not in pickup]

        solnTour = generateFeasiblePDTour(dimension, pickup, delivery, randomSeeds[i])

        coordinates = []
        for loc in range(dimension):
            randCoord = [random.randint(0,200) for _ in range(2)]
            while randCoord in coordinates:
                # avoid duplicates
                randCoord = [random.randint(0,200) for _ in range(2)]
            coordinates.append(randCoord)

        minTravelTime = []
        for j in range(dimension-1):
            minTravelTime.append(Distance_EUC_2D(coordinates[solnTour[j] - 1], coordinates[solnTour[j+1] - 1]))

        cumMinTravelTime = np.cumsum(minTravelTime)

        cost = cumMinTravelTime[-1] + Distance_EUC_2D(coordinates[solnTour[-1] - 1], coordinates[solnTour[0]]) # add time to return to depot

        maxTW = cumMinTravelTime[-1] + Distance_EUC_2D(coordinates[solnTour[dimension-1] - 1], coordinates[0]) \
            + random.randint(tw_min,tw_max) # set TW of depot to be large

        demand = [random.randint(0,50) for _ in range(int((dimension-1)/2))]
        minCapacity = []
        for j in range(dimension-1):
            loc = solnTour[j+1]
            if loc in pickup:
                minCapacity.append(demand[pickup.index(loc)])
            else:
                minCapacity.append(-demand[delivery.index(loc)])

        cumMinCapacity = np.cumsum(minCapacity)

        capacity = max(cumMinCapacity) + random.randint(0,200)

        lines.append(f'NAME : {name}')
        lines.append('TYPE : 1PDPTW')
        lines.append(f'DIMENSION : {dimension}')
        lines.append(f'VEHICLES : {vehicleNum}')
        lines.append(f'CAPACITY : {capacity}')
        lines.append('EDGE_WEIGHT_TYPE : EXACT_2D')

        lines.append('NODE_COORD_SECTION')
        for loc in range(dimension):
            lines.append(f'{loc+1} {coordinates[loc][0]} {coordinates[loc][1]}')

        lines.append('PICKUP_AND_DELIVERY_SECTION')
        lines.append(f'1 0 0 {maxTW} 0 0 0') # create data for depot
        for loc in range(2, dimension+1):
            TWrange = random.randint(tw_min, tw_max)
            est = max(0, cumMinTravelTime[solnTour.index(loc) - 1] - TWrange)
            lft = cumMinTravelTime[solnTour.index(loc) - 1] + TWrange
            actualDemand = minCapacity[solnTour.index(loc) - 1]

            if loc in pickup:
                pickupLoc = 0
                deliveryLoc = delivery[pickup.index(loc)]
            else:
                pickupLoc = pickup[delivery.index(loc)]
                deliveryLoc = 0

            lines.append(f'{loc} {actualDemand} {est} {lft} {serviceTime} {pickupLoc} {deliveryLoc}')

        lines.append('DEPOT_SECTION')
        lines.append('1')
        lines.append('-1')
        lines.append('EOF')

        f = open("./data/1PDPTW_generated_d{}_i{}_tmin{}_tmax{}_sd{}{}/INSTANCES/{}.txt".format(dimension, numInstance, tw_min, tw_max, randSeed, ext, name), "w") # write instance file
        for l in lines:
            # print(l)
            f.write(l + '\n')
        f.close()

        f = open("./data/1PDPTW_generated_d{}_i{}_tmin{}_tmax{}_sd{}{}/TOURS/{}_feasible.txt".format(dimension, numInstance, tw_min, tw_max, randSeed, ext, name), "w") # write solution file
        f.write(str(cost) + '\n')
        f.write(' '.join(str(x) for x in solnTour))
        f.close()

        # solution check
        instance = read1PDPTW("./data/1PDPTW_generated_d{}_i{}_tmin{}_tmax{}_sd{}{}/INSTANCES/{}.txt".format(dimension, numInstance, tw_min, tw_max, randSeed, ext, name))
        cost, tour = read1PDPTW_tour("./data/1PDPTW_generated_d{}_i{}_tmin{}_tmax{}_sd{}{}/TOURS/{}_feasible.txt".format(dimension, numInstance, tw_min, tw_max, randSeed, ext, name))
        precedence_check, tw_check, capacity_check, error, violatedLoc, curTime = check1PDPTW(tour, instance)
        if len(error) != 0:
            print("Generated tour is infeasible", error)
            raise ValueError


#generate_1PDPTW(11, 1, 2022)
generate_1PDPTW(11, 100000, 2022, tw_min=300, tw_max=500, ext='')
