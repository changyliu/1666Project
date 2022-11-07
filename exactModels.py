import gurobipy as gp
from distance import Distance_EUC_2D
from dataProcess import read1PDPTW


def getDistanceMatrix(instance):
    distMatrix = []
    for i in range(instance['numLocation']):
        curRow = []
        for j in range(instance['numLocation']):
            curRow.append(Distance_EUC_2D(instance['coordinates'][i], instance['coordinates'][j]))
        distMatrix.append(curRow)
    
    return distMatrix

def solve1PDPTW_MIP(instance):

    # prep data
    M = 999999
    distMatrix = getDistanceMatrix(instance)
    V = range(instance['numLocation']) # set of vertices
    P = [loc - 1 for loc in instance['pickup'] if loc != 0]  # set of pickup locations
    D = [loc - 1 for loc in instance['delivery'] if loc != 0] # set of delivery locations

    MIP = gp.Model('MIP')

    # variables
    x = MIP.addVars(V, V, vtype = gp.GRB.BINARY, name = 'x') # x_ij = 1 is location j is visited after location i
    s = MIP.addVars(V, vtype = gp.GRB.INTEGER, lb = 0, name = 's') # time vehicle arrives to location
    q = MIP.addVars(V, vtype = gp.GRB.INTEGER, lb = 0, name = 'q') # load of vehicle when arriving to location

    # objective function: minimize the distance
    MIP.setObjective(gp.quicksum(distMatrix[i][j] * x[i,j] for i in V for j in V), gp.GRB.MINIMIZE) 

    # constraints

    # each location is only visited once
    for i in (P + D):
        MIP.addConstr(gp.quicksum(x[i,j] for j in V) == 1)
    for j in (P + D):
        MIP.addConstr(gp.quicksum(x[i,j] for i in V) == 1)

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
    print(MIP.check_optimization_results())

    soln = [1]
    curLoc = 1
    print(f'{curLoc}')
    for i in range(len(V) - 1):
        nextLoc = [int(x[curLoc,j].x) for j in V].index(1)
        print(f' --> {nextLoc}')
        soln.append(nextLoc)
        curLoc = nextLoc

    return soln


instance = read1PDPTW('data/1PDPTW_generated/INSTANCES/generated-11-0.txt')
# print(getDistanceMatrix(instance))
solve1PDPTW_MIP(instance)