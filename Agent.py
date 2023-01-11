from abc import ABC, abstractmethod

import os
import torch
import numpy as np

from dataProcess import read1PDPTW
from solnCheck import check1PDPTW, check1PDPTW_all
from models import init_model, gen_solution
from utils import get_best_model, dotdict, float_to_str, computeCost, get_static_state

from solnRepair import localSearchExtended, cplex_MIP
from adaptive_lns import ALNS_Solver
from heuristics import construction_heuristic

import time

import config as c
config = c.config()

class Agent(ABC):
    """
    Base solver class for 1PDPTW.

    """

    def __init__(self):
        pass

    @abstractmethod
    def solve(self):
        pass

    def get_status(self, instance, solution):
        # feasibility check
        # _soln = [x+1 for x in solution]

        p_check, tw_check, c_check, error, error_dict, curLoc, curTime = check1PDPTW_all(solution, instance, return_now=False)
        if len(error) == 0:
            return 'feasible'
        else:
            return 'infeasible'

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
        solution = construction_heuristic(instance)
        if self.args.heuristic_ls == True:
            solution, numIter, timeSpent, num_dict, _ = \
                localSearchExtended(solution, instance, 10000, 10, strategy = 2, verbose=0)
        else:
            numIter = np.NAN
            num_dict = np.NAN
        end = time.time()

        if len(solution) > 0:
            cost = computeCost(solution, instance)
        else:
            cost = np.NAN
        status = self.get_status(instance, solution)

        return solution, cost, end-start, status, numIter, num_dict

class ALNSAgent(Agent):
    def __init__(self, *args):
        super().__init__()
        self.args = args[0]

    def solve(self, instance):
        coords, _, _, _, _, W, E, L = get_static_state(instance)

        start = time.time()
        alns_solver = ALNS_Solver(
                        instance, 
                        degree_of_destruction=self.args.degree_of_destruction, 
                        epsilon=self.args.epsilon,
                        beta=self.args.beta_alns,
                        cost_func=self.args.cost_func_alns, 
                        seed=self.args.seed
                        )
        alns_solver.build()
        solution = alns_solver.solve(iterations = 15000)
        solution = [x+1 for x in solution]
        end = time.time()

        cost = computeCost(solution, instance)
        status = self.get_status(instance, solution)

        numIter = np.NAN
        num_dict = np.NAN

        return solution, cost, end-start, status, numIter, num_dict

class RLAgent(Agent):
    def __init__(self, *args, model_dir, device=torch.device('cpu')):
        super().__init__()
        self.model_dir = model_dir
        self.device = device
        self.args = args[0]

        best_fname, _ = get_best_model(model_dir)

        #Load checkpoint
        self.Q_func, _, _, _ = init_model(
                                    self.args, 
                                    model_name='sample',
                                    fname=os.path.join(model_dir, best_fname), 
                                    device=device
                                    )
        print("Successfully loaded a model")

    def solve(self, instance):
        start = time.time()
        solution, coords, W, E, L = gen_solution(
                                        self.Q_func, 
                                        instance, 
                                        device=self.device
                                        )
        end = time.time()
        solution = [x+1 for x in solution]

        cost = computeCost(solution, instance)
        status = self.get_status(instance, solution)
        numIter = np.NAN
        num_dict = np.NAN
        
        return solution, cost, end-start, status, numIter, num_dict

class RLAgent_repair(RLAgent):
    def __init__(self, *args, model_dir, device=torch.device('cpu')):
        super().__init__(args[0], model_dir=model_dir, device=device)
    
    def solve(self, instance):
        start = time.time()
        solution, coords, W, E, L = gen_solution(
                                        self.Q_func, 
                                        instance, 
                                        device=self.device
                                        )
        # print("solution (before): ", solution)
        #precedence_check, tw_check, capacity_check, error, violatedLoc, locTime = check1PDPTW(solution, instance, return_now=False)
        #print(precedence_check, tw_check, capacity_check, error, violatedLoc, locTime)

        if self.args.repair == 'ls':
            # local search
            solution, numIter, timeSpent, num_dict, feasible = localSearchExtended([x+1 for x in solution], instance, 10000, 10, strategy = self.args.repair_strategy)
            # solution = [x-1 for x in solution]
        elif self.args.repair == 'mip_cplex':
            # using cplex
            solution, cost, timeSpent = cplex_MIP([x+1 for x in solution], instance, 10000, 10, verbose=0)
            # solution = [x-1 for x in solution]
            numIter = np.NAN
            num_dict = np.NAN
        elif self.args.repair == 'alns':
            alns_solver = ALNS_Solver(
                            instance, 
                            degree_of_destruction=self.args.degree_of_destruction, 
                            epsilon=self.args.epsilon,
                            beta=self.args.beta_alns,
                            cost_func='tw', # alns-repair always minimizes tw violation only
                            verbose=0
                            )
            alns_solver.build()
            solution = alns_solver.resume(tour=solution, iterations = 15000)
            solution = [x+1 for x in solution]
            numIter = np.NAN
            num_dict = np.NAN
        else:
            raise NotImplementedError

        end = time.time()
        #print("solution (after): ", solution, feasible)
        #precedence_check, tw_check, capacity_check, error, violatedLoc, locTime = check1PDPTW(solution, instance, return_now=False)
        #print(precedence_check, tw_check, capacity_check, error, violatedLoc, locTime)

        if len(solution) > 0:
            cost = computeCost(solution, instance)
        else:
            cost = np.NAN
        status = self.get_status(instance, solution)
        return solution, cost, end-start, status, numIter, num_dict

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args = dotdict({
        'dataset_name': '1PDPTW_generated_d15_i100000_tmin300_tmax500_sd2022',
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

    model_name = '{}_ed{}_ne{}_bs{}_lr{}_bt{}_sd{}'.format(
                    args.dataset_name,
                    args.emb_dim,
                    args.num_episodes,
                    args.batch_size,
                    float_to_str(args.lr),
                    args.beta,
                    args.seed
                    )
    model_dir = os.path.join('.', config['MODEL_DIR'], model_name)

    #agent = RLAgent_repair(args, model_dir=model_dir, device=device)
    agent = HeuristicAgent(args)
    ins_num = 74
    instance = read1PDPTW('data/1PDPTW_generated_d31_i200_tmin300_tmax500_sd2022_test/INSTANCES/generated-{}.txt'.format(ins_num))
    solution, cost, solve_time, status, numIter, num_dict = agent.solve(instance)
    print(solution, cost, solve_time, status, numIter, num_dict)