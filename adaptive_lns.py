import statistics
from ALNS.alns import ALNS, State
from ALNS.alns.criteria import HillClimbing, SimulatedAnnealing

import copy
import itertools
import random

import numpy as np

import networkx as nx

import tsplib95
import tsplib95.distances as distances

import matplotlib.pyplot as plt

from copy import deepcopy

from dataProcess import read1PDPTW, read1PDPTW_tour
from utils import total_distance, get_static_state, cost_func

SEED = 9876

class TspState(State):
    """
    Solution class for the TSP problem. It has two data members, nodes, and edges.
    nodes is a list of node tuples: (id, coords). The edges data member, then, is
    a mapping from each node to their only outgoing node.
    """

    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def copy(self):
        return deepcopy(self)

    def objective(self):
        """
        The objective function is simply the sum of all individual edge lengths,
        using the rounded Euclidean norm.
        """
        return sum(distances.euclidean(node[1], self.edges[node][1])
                   for node in self.nodes)
    
    def to_graph(self):
        """
        NetworkX helper method.
        """
        graph = nx.Graph()

        for node, coord in self.nodes:
            graph.add_node(node, pos=coord)

        for node_from, node_to in self.edges.items():
            graph.add_edge(node_from[0], node_to[0])

        return graph

class TspState_dist(State):
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def copy(self):
        return deepcopy(self)

    def objective(self):
        """
        The objective function is simply the sum of all individual edge lengths.
        """
        #print(self.nodes)
        #print(self.edges)

        return sum([node[1][self.edges[node][0]] for node in self.nodes])

class TspState_v2(State):
    def __init__(self, coords, W, solution, capacity, demands, pickup, delivery, E, L, beta=10, cost_func='tw'):
        self.coords = coords
        self.W = W
        self.solution = solution

        self.capacity = capacity
        self.demands = demands
        self.pickup = pickup
        self.delivery = delivery

        self.load = 0

        self.E = E # ealiest arrival time
        self.L = L # latest arrival time

        self.beta = beta # penalty factor
        self.cost_func = cost_func # 'tw' if we only minimize tw violation

    def copy(self):
        return deepcopy(self)

    def objective(self):
        if -1 in self.solution: # incomplete solution
            return float('inf')
        return cost_func(self.solution, self.W, self.E, self.L, beta=self.beta, mode=self.cost_func)
    
    def time_window_violation(self):
        return time_window_violation(self)
    
def time_window_violation(state):
    violation = [0]
    t = 0 # arrival time
    for i in range(len(state.solution) - 1):
        delta_t = state.W[state.solution[i], state.solution[i+1]]
        t += delta_t

        e = state.E[state.solution[i+1]]
        l = state.L[state.solution[i+1]]

        if t - l > 0:
            violation.append(t - l)
        elif e - t > 0:
            violation.append(t - e)
        else:
            violation.append(0)

    return violation

class ALNS_Solver():
    def __init__(self, instance, degree_of_destruction=0.25, criterion='HillClimbing', 
                    epsilon=0.05, beta=10, early_stopping=True, seed=0, cost_func='tw', verbose=0):
        self.instance = instance
        self.degree_of_destruction = degree_of_destruction
        self.epsilon = epsilon # epsilon greedy
        self.beta = beta # penalty factor for tw violation

        self.early_stopping = early_stopping # stop the algorithm if no improvement is made for a while
        self.stop_iter = 10000 # stop if no improvement is made for 10000 iterations

        self.cost_func = cost_func # if 'tw', it only minimizes time window violations

        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.verbose = verbose

        if criterion=='HillClimbing':
            self.criterion = HillClimbing()
        elif criterion=='SimulatedAnnealing':
            self.criterion = SimulatedAnnealing()

    def pairs_to_remove(self, state):
        return int((len(state.coords) - 1) * self.degree_of_destruction / 2)

    def worst_removal(self, current, random_state):
        """
        Worst removal iteratively removes the 'worst' edges, that is,
        those edges that have the worst contribution to the objective function.

        That is, the nodes are sorted by
            time window violation + distance to the former node + distance to its subsequent node
        and destroy the nodes in this order.

        """
        destroyed = current.copy()

        violation = time_window_violation(destroyed) # time window violation
        N = len(destroyed.coords)

        dists = []
        for i in range(N):
            if i != 0:
                dist_pre = destroyed.W[destroyed.solution[i-1], destroyed.solution[i]]
            if i != (N - 1):
                dist_post = destroyed.W[destroyed.solution[i], destroyed.solution[i+1]]
            
            if i == 0:
                d = dist_post
            elif i != (N - 1):
                d = dist_pre
            else:
                d = np.mean([dist_pre, dist_post])            
            dists.append(d)
        
        penalty = [self.beta * abs(violation[i]) + dists[i] for i in range(len(destroyed.coords))]
        node_idxs = np.argsort(penalty)[::-1].tolist()
        node_idxs = [x for x in node_idxs if x != 0]

        # remove the nodes and their corresponding nodes in the order of node_idxs
        for i in range(self.pairs_to_remove(current)):
            node_idx = node_idxs.pop(0)
            while destroyed.solution[node_idx] == -1:
                node_idx = node_idxs.pop(0)
            node = destroyed.solution[node_idx]

            # destroy the chosen node
            destroyed.solution[node_idx] = -1

            # choose the corresponding pair of the chosen node
            if destroyed.pickup[node] != 0:
                paired = destroyed.pickup[node] - 1
            else:
                paired = destroyed.delivery[node] - 1
            pos = destroyed.solution.index(paired)
            # destroy the corresponding pair node
            destroyed.solution[pos] = -1

        return destroyed

    def path_removal(self, current, random_state):
        """
        Removes an entire consecutive subpath, that is, a series of
        contiguous edges.

        1) Choose the start node of the path
        2) Remove the nodes and their corresponding pair nodes (delivery node 
           if the chosen node is pickup and vice versa) until the number of destroyed
           nodes reaches a certain number determined by degree_of_destruction 

        """
        destroyed = current.copy()
        pair_num = self.pairs_to_remove(current)
        # latest possible start node of a path
        max_idx = len(destroyed.coords) - 2 * pair_num

        # choose the start node
        candidates = [i for i in range(1, max_idx + 1)]
        node_idx = np.random.choice(candidates) # start of the path
        node = destroyed.solution[node_idx]

        for i in range(pair_num):
            # proceed the index until it hits an unbroken node
            while destroyed.solution[node_idx] == -1:
                node_idx += 1
            node = destroyed.solution[node_idx]

            # destroy the chosen node
            destroyed.solution[node_idx] = -1

            # choose the corresponding pair of the chosen node
            if destroyed.pickup[node] != 0:
                paired = destroyed.pickup[node] - 1
            else:
                paired = destroyed.delivery[node] - 1
            pos = destroyed.solution.index(paired)
            # destroy the corresponding pair node
            destroyed.solution[pos] = -1

            node_idx += 1

        return destroyed

    def random_removal(self, current, random_state):
        """
        Random removal iteratively removes random edges.
        """
        destroyed = current.copy()

        delivery = [i for i in range(len(destroyed.coords)) if destroyed.pickup[i] != 0]
        chosen = np.random.choice(delivery, self.pairs_to_remove(current), replace=False)
        for d in chosen:
            pos_d = destroyed.solution.index(d)
            destroyed.solution[pos_d] = -1

            # corresponding pickup node
            p = destroyed.pickup[d] - 1
            pos_p = destroyed.solution.index(p)
            destroyed.solution[pos_p] = -1

        return destroyed

    def would_form_subcycle(self, from_node, to_node, state):
        """
        Ensures the proposed solution would not result in a cycle smaller
        than the entire set of nodes. Notice the offsets: we do not count
        the current node under consideration, as it cannot yet be part of
        a cycle.
        """
        for step in range(1, len(state.nodes)):
            if to_node not in state.edges:
                return False

            to_node = state.edges[to_node]
            
            if from_node == to_node and step != len(state.nodes) - 1:
                return True

        return False

    def compute_cost(self, current, t, cur, ne):
        """
        Compute the cost increase by going from the current node 
        in the partial solution to next_node.
        
        current (State): the current state.
        t (int)        : the current time.
        cur (int)      : the current node.
        ne (int)       : the node that will be visited from the current node.

        """
        a = t + current.W[cur, ne] # actual arrival time
        tw_violation = self.beta * (max(a-current.L[ne], 0) + max(current.E[ne]-a, 0))
        if self.cost_func == 'tw':
            return tw_violation
        else:
            return current.W[cur, ne] + tw_violation

    def get_last_unbroken_node(self, current):
        """
        Get the last unbroken node in the first subpath in the partial solution.
        e.g., [3, 5, -1, -1, 2] --> cur_idx = 1, cur_node = 5

        """
        if -1 not in current.solution:
            return -1, -1

        cur_idx = [i for i in range(len(current.coords))
                    if current.solution[i] == -1][0] - 1
        cur_node = current.solution[cur_idx]
        return cur_idx, cur_node

    def greedy_repair(self, current, random_state):
        """
        Greedily repairs a tour, stitching up nodes that are not departed
        with those not visited.
        """

        if -1 not in current.solution:
            # solution is already complete
            return current
        
        assert current.solution[0] == 0, "start is not depot. solution={}".format(current.solution) # starts from depot
        cur_idx = 0
        cur_node = current.solution[cur_idx]
        current.load = 0

        # current time
        t = 0
        while -1 in current.solution:
            if current.solution[cur_idx+1] == -1: # unbroken
                # Feasible set for the next node. Node i is feasible if:
                #1) there's a path from the last node in the partial solution to i
                #2) i is not in the partial solution
                #3) if it is a delivery node, pick up is done
                #4) load will not exceed the capacity.
                # print(current.W)
                feasible = [i for i in range(len(current.coords))
                            if len(current.solution) == 0 or current.W[cur_node, i] > 0
                            if i not in current.solution
                            if current.pickup[i] == 0 or (current.pickup[i] - 1) in current.solution
                            if current.load + current.demands[i] <= current.capacity]
                if len(feasible) == 0: # search failed
                    return current

                objVals = [self.compute_cost(current, t, cur_node, f) for f in feasible]

                # Choose the next node epsilon greedily
                if np.random.rand() < self.epsilon:
                    greedy_best = np.random.choice(feasible)
                else:
                    greedy_best = feasible[np.argmin(objVals)]

                current.solution[cur_idx + 1] = greedy_best

            t += current.W[cur_node, current.solution[cur_idx+1]]
            cur_idx += 1
            cur_node = current.solution[cur_idx]
            current.load += current.demands[cur_node]

            if current.load > current.capacity:
                # make it an incomplete solution so that it is rejected by LNS
                current.solution[cur_idx] = -1
                return current

        return current
    
    def build(self):
        coords, capacity, demands, pickup, delivery, W, E, L = get_static_state(self.instance)

        current = [-1 if i != 0 else 0 for i in range(len(coords))]
        self.initial_solution = TspState_v2(
                                    coords, W, current, capacity, demands, pickup, delivery, 
                                    E, L, beta=self.beta, cost_func=self.cost_func
                                    )

        while -1 in self.initial_solution.solution:
            current = [-1 if i != 0 else 0 for i in range(len(coords))]
            self.state = TspState_v2(
                            coords, W, current, capacity, demands, pickup, delivery, 
                            E, L, beta=self.beta, cost_func=self.cost_func
                            )
            self.random_state = np.random.RandomState(self.seed)
            self.initial_solution = self.greedy_repair(self.state, self.random_state)

        total_dist = total_distance(self.initial_solution.solution, W)
        if self.verbose:
            print("Initial solution is {0}.".format(self.initial_solution.solution))
            print("Initial solution objective is {0}.".format(self.initial_solution.objective()))
            print("Total distance is {0}".format(total_dist))
            print("TW violation: ", self.initial_solution.objective() - total_dist)

        self.alns = ALNS(self.random_state)

        self.alns.add_destroy_operator(self.random_removal)
        self.alns.add_destroy_operator(self.path_removal)
        self.alns.add_destroy_operator(self.worst_removal)

        self.alns.add_repair_operator(self.greedy_repair)
    
    def solve(self, time_limit=10, iterations=5000, tour=None):
        if tour is not None:
            self.solution_sharing(tour)

        result = self.alns.iterate(self.initial_solution, [3, 2, 1, 0.5], 0.8, self.criterion,
                      iterations=iterations, collect_stats=True, time_limit=time_limit, 
                      early_stopping=self.early_stopping, stop_iter=self.stop_iter, verbose=self.verbose)
        self.cur_state = result.best_state

        return self.cur_state.solution
    
    def resume(self, time_limit=10, iterations=5000, tour=None):
        return self.solve(time_limit=time_limit, iterations=iterations, tour=tour)

    def solution_sharing(self, tour):
        self.initial_solution.solution = tour

if __name__ == "__main__":
    ins_num = 57
    print("instance", ins_num)
    instance = read1PDPTW('data/1PDPTW_generated_d15_i1000_tmin300_tmax500_sd2022_test/INSTANCES/generated-{}.txt'.format(ins_num))
    alns_solver = ALNS_Solver(instance, seed=0, beta=10, cost_func='tw', degree_of_destruction=0.6, epsilon=0.05, early_stopping=False, verbose=1)
    alns_solver.build()
    alns_solver.solve(iterations=20000)
    print("")

    #tour = [0, 4, 5, 1, 2, 9, 7, 3, 10, 6, 8]
    #alns_agent.solution_sharing(tour)
    #destroyed = alns_agent.worst_removal(alns_agent.initial_solution, np.random.RandomState(9876))
    #print(destroyed.solution)

    #result = alns_solver.resume(time_limit=30, iterations=15000, tour=tour)
    #print(result)

    #solution = result.statistics['solution']
    #print("solution: ", solution)

    #objective = solution.objective()

    #print('Best heuristic objective is {0}.'.format(objective))