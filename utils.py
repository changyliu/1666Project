import numpy as np
import os
from scipy.spatial import distance_matrix
import math

import config as c
config = c.config()

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

def get_best_model(model_dir):
    """
    Get file with smallest cost in the model directory.

    model_dir (str) : path to the model directory
    
    """

    all_lengths_fnames = [f for f in os.listdir(model_dir) if f.endswith('.tar')]
    best_fname = sorted(all_lengths_fnames, key=lambda s: float(s.split('.tar')[0].split('_')[-1]))[0]
    cost = best_fname.split('.tar')[0].split('_')[-1]
    return best_fname, cost

def float_to_str(val):
    """
    e.x) 0.5 --> '05'
    
    """

    return str(val).replace('.', '')

def moving_avg(x, N=10):
    return np.convolve(np.array(x), np.ones((N,))/N, mode='valid')

def cost_func(solution, W, entering_times, leaving_times, mode='tw', beta=1):
    """
    Cost function for PDPTW.

    W (tensor) : distance matrix (note that it must be tensor!)
    beta (int) : penalty factor of violating the time window constraint.    
    
    """

    if len(solution) < 2:
        return 0

    penalty = []
    a = 0 # arrival time
    for i in range(len(solution) - 1):
        a += W[solution[i], solution[i+1]].item()
        e = entering_times[solution[i+1]]
        l = leaving_times[solution[i+1]]
        #if arriving_time < e:
        #    arriving_time = e
        
        penalty.append(max(a-l, 0) + max(e-a, 0))

    if mode == 'tw':
        return beta * sum(penalty)
    else:
        return total_distance(solution, W) + beta * sum(penalty)

def total_distance(solution, W):
    if len(solution) < 2:
        return 0  # there is no travel
    
    total_dist = 0
    for i in range(len(solution) - 1):
        total_dist += W[solution[i], solution[i+1]].item()
        
    # if this solution is "complete", go back to initial point
    if len(solution) == W.shape[0]:
        total_dist += W[solution[-1], solution[0]].item()

    return total_dist

def get_static_state(instance):
    """
    Given an instance dict, return the static state for RL.

    coords (ndarray): city coordinates
    capacity (int)  : capacity of the vehicle
    demands (list)  : list of the demand of each location
    pickup (list)   : list of pickup locations (0 for depot and pickup nodes)
    delivery (list) : list of delivery locations (0 for depot and delivery nodes)
    W (ndarray)     : distance matrix
    E (list)        : list of entering times
    L (list)        : list of leaving times

    """
    assert type(instance) == dict
    if instance['type'] == "1PDPTW":
        coords = np.array(instance['coordinates'])
        capacity = instance['capacity']
        demands = instance['demand']
        pickup = instance['pickup']
        delivery = instance['delivery']
        W = distance_matrix(coords, coords)

        E = []
        L = []
        for (e, l) in instance['tw']:
            E.append(e)
            L.append(l)

        return coords, capacity, demands, pickup, delivery, W, E, L
    else:
        raise NotImplementedError


def Distance_EUC_2D(locA, locB):
    xd = locA[0] - locB[0]
    yd = locA[1] - locB[1]
    return int(math.sqrt(xd * xd + yd * yd))


def computeCost(soln, instance):
    totalTravelTime = 0
    for i in range(instance['numLocation']-1):
        totalTravelTime += Distance_EUC_2D(instance['coordinates'][soln[i] - 1], instance['coordinates'][soln[i+1] - 1])

    totalTravelTime += Distance_EUC_2D(instance['coordinates'][soln[-1] - 1], instance['coordinates'][soln[0]]) # add time to return to depot
    return totalTravelTime

def getDistanceMatrix(instance):
    distMatrix = []
    instance['coordinates'].append(instance['coordinates'][0]) # add coordinates of artificial ending depot
    for i in range(instance['numLocation'] + 1):
        curRow = []
        for j in range(instance['numLocation'] + 1):
            curRow.append(Distance_EUC_2D(instance['coordinates'][i], instance['coordinates'][j]))
        distMatrix.append(curRow)
    
    return distMatrix

def get_x_from_soln(soln):
    x = np.zeros((len(soln)+1, len(soln)+1))

    for i in range(len(soln)-1):
        x[soln[i]-1][soln[i+1]-1] = 1
    
    return x

# print(get_x_from_soln([1, 4, 5, 9, 10, 6, 8, 7, 2, 11, 3]))