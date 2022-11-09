import gurobipy as gp
from distance import Distance_EUC_2D
from dataProcess import read1PDPTW


def getDistanceMatrix(instance):
    distMatrix = []
    instance['coordinates'].append(instance['coordinates'][0]) # add coordinates of artificial ending depot
    for i in range(instance['numLocation'] + 1):
        curRow = []
        for j in range(instance['numLocation'] + 1):
            curRow.append(Distance_EUC_2D(instance['coordinates'][i], instance['coordinates'][j]))
        distMatrix.append(curRow)
    
    return distMatrix

def solve1PDPTW_MIP(instance):

    # prep data
    M = 999999
    distMatrix = getDistanceMatrix(instance)
    V = range(instance['numLocation'] + 1) # set of vertices, create extra vertex for returning to depot
    P = [loc - 1 for loc in instance['pickup'] if loc != 0]  # set of pickup locations
    D = [loc - 1 for loc in instance['delivery'] if loc != 0] # set of delivery locations

    instance['demand'].append(0) # add 0 demand for artificial ending depot
    instance['tw'].append(instance['tw'][0]) # add tw for artificial ending depot

    MIP = gp.Model('MIP')

    # variables
    x = MIP.addVars(V, V, vtype = gp.GRB.BINARY, name = 'x') # x_ij = 1 is location j is visited after location i
    s = MIP.addVars(V, vtype = gp.GRB.INTEGER, lb = 0, name = 's') # time vehicle arrives to location
    q = MIP.addVars(V, vtype = gp.GRB.INTEGER, lb = 0, name = 'q') # load of vehicle when arriving to location

    # objective function: minimize the distance
    MIP.setObjective(gp.quicksum(distMatrix[i][j] * x[i,j] for i in V for j in V), gp.GRB.MINIMIZE) 

    # constraints

    # each location is only visited once
    for i in (P + D + [0]): # out
        MIP.addConstr(gp.quicksum(x[i,j] for j in (P + D + [len(V) - 1]) if i != j) == 1)
    for j in (P + D + [len(V) - 1]): # in
        MIP.addConstr(gp.quicksum(x[i,j] for i in (P + D + [0]) if i != j) == 1)
    
    # # start and end at depot
    # MIP.addConstr(gp.quicksum(x[0,j] for j in (P + D + [len(V) - 1])) == 1)
    # MIP.addConstr(gp.quicksum(x[i,len(V) - 1] for i in (P + D + [0])) == 1)

    # define s and q
    for i in V:
        for j in V:
            MIP.addConstr(s[j] >= s[i] + distMatrix[i][j] - M * ( 1- x[i,j]))
            MIP.addConstr(q[j] >= q[i] + instance['demand'][i] - M * ( 1- x[i,j]))
    
    # tw
    for i in V:
        MIP.addConstr(instance['tw'][i][0] <= s[i])
        MIP.addConstr(s[i] <= instance['tw'][i][1])

    # capacity
    for i in V:
        MIP.addConstr(q[i] <= instance['capacity'])

    # precedence
    for i in P:
        MIP.addConstr(s[instance['delivery'][i] - 1] >= s[i] + distMatrix[i][instance['delivery'][i] - 1])
   
    # optimizing
    MIP.optimize()

    # get results

    soln = [0 + 1]
    curLoc = 0
    route = f'{0 + 1}'
    for i in range(len(V) - 1):
        # print([int(x[curLoc,j].x) for j in V])
        nextLoc = [int(x[curLoc,j].x) for j in V].index(1)
        route += (f' -> {nextLoc + 1}')
        soln.append(nextLoc + 1)
        curLoc = nextLoc
    
    cost = MIP.ObjVal

    print('\n')
    print(soln)
    print(f'Route: {route}')
    print(f'Cost: {cost}')

    return soln[0:-1], cost


instance = read1PDPTW('data/1PDPTW_generated_test/INSTANCES/generated-2.txt')
# print(getDistanceMatrix(instance))
solve1PDPTW_MIP(instance)