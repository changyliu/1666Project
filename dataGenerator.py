import numpy as np
import random
import os
from distance import Distance_EUC_2D
from solnGenerator import generateFeasiblePDTour

def generate_1PDPTW(dimension, numInstance, randSeed):
    if not os.path.exists(f"data/1PDPTW_generated"):
        os.makedirs(f"data/1PDPTW_generated")
    os.mkdir(f"data/1PDPTW_generated/INSTANCES/")
    os.mkdir(f"data/1PDPTW_generated/TOURS/")

    vehicleNum = 1
    serviceTime = 0
    
    random.seed(randSeed)

    randomSeeds = [random.randint(0,5000) for i in range(numInstance)]

    for i in range(numInstance):

        random.seed(randomSeeds[i])

        lines = []

        name = f'generated-{dimension}-{i}'
        capacity = random.randint(100,500)

        pickup = random.sample(range(2, dimension+1), int((dimension-1)/2)) # 1 is depot location, no pickup or delivery
        delivery = [x for x in range(2, dimension+1) if x not in pickup]

        solnTour = generateFeasiblePDTour(dimension, pickup, delivery, 2022)

        coordinates = []
        for loc in range(dimension):
            randCoord = [random.randint(0,200) for i in range(2)]
            coordinates.append(randCoord)

        minTravelTime = []
        for i in range(dimension-1):
            minTravelTime.append(Distance_EUC_2D(coordinates[solnTour[i] - 1], coordinates[solnTour[i+1] - 1]))
        
        cumMinTravelTime = np.cumsum(minTravelTime)

        maxTW = cumMinTravelTime[-1] + Distance_EUC_2D(coordinates[solnTour[dimension-1] - 1], coordinates[0]) \
            + random.randint(100,300) # set TW of depot to be large

        demand = [random.randint(0,50) for i in range(int((dimension-1)/2))]
        minCapacity = []
        for i in range(dimension-1):
            loc = solnTour[i+1]
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
            TWrange = random.randint(100,300)
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

        f = open(f"data/1PDPTW_generated/INSTANCES/{name}.txt", "w") # write instance file
        for l in lines:
            # print(l)
            f.write(l + '\n')
        f.close()

        f = open(f"data/1PDPTW_generated/TOURS/{name}.txt", "w") # write solution file
        f.write(' '.join(str(x) for x in solnTour))
        f.close()


generate_1PDPTW(11, 10, 2022)