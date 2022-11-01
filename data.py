# Read and Process Data

'''
Pickup-and-delivery problem with time windows (PDPTW)

The subdirectory INSTANCES contains the following benchmark instances:

	Li (356 instances)
	Source:
	Li, H., Lim, A.:
	A Metaheuristic for the Pickup and Delivery Problem with Time Windows.
	Proceedings of the 13th IEEE International Conference on Tools with
	Artificial Intelligence:160-167 (2001)

	Ropke (42 instances)
	Source:
	Ropke, S., Pisinger, D.:
	An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery 
	Problem with Time Windows.
	Transp. Sci., 40(4):455-472 (2006)
	
PICKUP_AND_DELIVERY_SECTION :
Each line is of the form

      <integer> <integer> <real> <real> <real> <integer> <integer>
 
The first integer gives the number of the node.
The second integer gives its demand.
The third and fourth number give the earliest and latest time for the node.
The fifth number specifies the service time for the node.
The last two integers are used to specify pickup and delivery. The first of 
these integers gives the index of the pickup sibling, whereas the second integer
gives the index of the delivery sibling.

The subdirectory TOURS contains the best tours found by LKH-3.

Tabulated results can be found in the subdirectory RESULTS.

NAME : LC1_2_1
TYPE : PDPTW
DIMENSION : 213
VEHICLES : 20
CAPACITY : 200
EDGE_WEIGHT_TYPE : EXACT_2D
NODE_COORD_SECTION
1 70 70
PICKUP_AND_DELIVERY_SECTION
1 0 0 1351 0 0 0
2 -20 750 809 90 72 0
'''

from os import listdir
from os.path import isfile, join


def readPDPTW(file_path):
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
        
