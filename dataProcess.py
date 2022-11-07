# Read and Process Data

from os.path import isfile, join


def read1PDPTW(file_path):
    f = open(file_path, "r")

    instance = {}
    instance['name'] = f.readline().split()[-1]
    instance['type'] = f.readline().split()[-1]
    instance['numLocation'] = int(f.readline().split()[-1])
    instance['numVehicle'] = int(f.readline().split()[-1])
    instance['capacity'] = int(f.readline().split()[-1])
    instance['egdeWeightType'] = f.readline().split()[-1]

    f.readline() #skip title line NODE_COORD_SECTION
    coord = []
    for i in range(instance['numLocation']):
        line = f.readline().split()
        coord.append((int(line[1]), int(line[2])))
    instance['coordinates'] = coord

    demand = []
    tw = []
    serviceTime = []
    pickup = []
    delivery = []
    f.readline() #skip title line PICKUP_AND_DELIVERY_SECTION
    for i in range(instance['numLocation']):
        line = f.readline().split()
        demand.append(int(line[1]))
        tw.append((int(line[2]), int(line[3])))
        serviceTime.append(int(line[4]))
        pickup.append(int(line[5]))
        delivery.append(int(line[6]))
    instance['demand'] = demand
    instance['tw'] = tw
    instance['serviceTime'] = serviceTime
    instance['pickup'] = pickup
    instance['delivery'] = delivery

    return instance
        
# instance = read1PDPTW('data/1PDPTW_generated/INSTANCES/generated-11-0.txt')
# print(instance['name'])