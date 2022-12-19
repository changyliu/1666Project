from dataProcess import read1PDPTW
from solnCheck import check1PDPTW_all
from utils import computeCost, dotdict

from solnRepair import localSearchExtended

from Agent import Agent
import time

import numpy as np

class HeuristicAgent(Agent):
    """
    Construction heuristic with a local search atop.

    Based on the following paper with using our local search instead of tabu search:
    https://link.springer.com/article/10.1023/A:1012204504849
    
    """

    def __init__(self, *args):
        super().__init__()
        self.args = args[0]

    def solve(self, instance):
        start = time.time()
        solution = self.construction_heuristic(instance)
        if self.args.heuristic_ls == True:
            solution, numIter, timeSpent, num_dict, _ = \
                localSearchExtended(solution, instance, 10000, 600, strategy = 2, verbose=0)
        else:
            numIter = np.NAN
            num_dict = np.NAN
        end = time.time()

        cost = computeCost(solution, instance)
        status = self.get_status(instance, solution)

        return solution, cost, end-start, status, numIter, num_dict

    def construction_heuristic(self, instance):
        # get the pickup indices
        pickup = [x-1 for x in instance['pickup'] if x != 0]
        pickup.sort()

        # sort pickup requests in the ascending order of est
        # tie-break by choosing earlier latest arrival time
        est = [instance['tw'][i][0] for i in pickup]
        lst = [instance['tw'][i][1] for i in pickup]
        idxs = [i for i in range(len(est))]

        sorted = []
        while len(est) > 0:
            cands = [i for i, x in enumerate(est) if x == min(est)]
            if len(cands) > 1: # tie-break
                ls = [lst[i] for i in cands]
                _idx = np.argmin(ls)
                idx = cands[_idx]
            else:
                idx = cands[0]

            sorted.append(pickup[idxs[idx]])
            est.pop(idx)
            lst.pop(idx)
            idxs.pop(idx)

        # insert the delivery node right after each pickup node
        tour = [0] # start from depot
        for p in sorted:
            tour.append(p)
            d = instance['delivery'][p]-1 # delivery request
            tour.append(d)
        tour = [x+1 for x in tour]

        return tour


if __name__ == "__main__":
    args = dotdict({
        'dataset_name': '1PDPTW_generated_d15_i1000_tmin300_tmax500_sd2022_test',
        'emb_dim'              : 20,
        'emb_iter_T'           : 1,
        'num_episodes'         : 30001,
        'batch_size'           : 32,
        'lr'                   : 5e-3,
        'lr_decay_rate'        : 1. - 2e-5,
        'beta'                 : 1,
        'repair'               : 'alns',
        'repair_strategy'      : 0,
        'beta_alns'            : 10,
        'epsilon'              : 0.05,
        'degree_of_destruction': 0.4,
        'cost_func_alns'       : 'all',
        'heuristic_ls'         : True,

        'seed'                 : 2
    })

    agent = HeuristicAgent(args)

    for ins_num in range(10):
        instance = read1PDPTW(f'data/1PDPTW_generated_d15_i1000_tmin300_tmax500_sd2022_test/INSTANCES/generated-{ins_num}.txt')
        solution, cost, solve_time, status, numIter, num_dict = agent.solve(instance)
        p_check, tw_check, c_check, error, error_dict, curLoc, curTime = check1PDPTW_all(solution, instance, return_now=False)
        print(f"Instance {ins_num}: {cost}, {solve_time}, {len(error)}, {p_check}, {tw_check}, {c_check}")
