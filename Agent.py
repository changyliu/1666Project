from abc import ABC, abstractmethod

import os
import torch

from dataProcess import read1PDPTW
from solnCheck import check1PDPTW, check1PDPTW_all
from models import init_model, gen_solution
from utils import get_best_model, dotdict, float_to_str, cost_func, get_static_state

from solnRepair import localSearchExtended, cplex_MIP
from adaptive_lns import ALNS_Solver

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
        _soln = [x+1 for x in solution]
        p_check, tw_check, c_check, error, error_dict, curLoc, curTime = check1PDPTW_all(_soln, instance, return_now=False)
        if len(error) == 0:
            return 'feasible'
        else:
            return 'infeasible'

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
                        seed=self.args.seed
                        )
        alns_solver.build()
        solution = alns_solver.solve(iterations = 15000)
        end = time.time()

        cost = cost_func(solution, W, E, L, beta=self.args.beta)
        status = self.get_status(instance, solution)
        return solution, cost, end-start, status

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

        cost = cost_func(solution, W, E, L, beta=self.args.beta)
        status = self.get_status(instance, solution)
        return solution, cost, end-start, status

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

        #print("solution (before): ", solution)
        #precedence_check, tw_check, capacity_check, error, violatedLoc, locTime = check1PDPTW(solution, instance, return_now=False)
        #print(precedence_check, tw_check, capacity_check, error, violatedLoc, locTime)

        if self.args.repair == 'ls':
            # local search
            solution, numIter, timeSpent, feasible = localSearchExtended([x+1 for x in solution], instance, 5000, 600, strategy = self.args.repair_strategy)
            solution = [x-1 for x in solution]
        elif self.args.repair == 'mip_cplex':
            # using cplex
            solution, cost, timeSpent = cplex_MIP(rl_soln, instance, iterLimit, timeLimit, verbose=0):
            solution = [x-1 for x in solution]
        elif self.args.repair == 'alns':
            alns_solver = ALNS_Solver(
                            instance, 
                            degree_of_destruction=self.args.degree_of_destruction, 
                            epsilon=self.args.epsilon,
                            beta=self.args.beta_alns
                            )
            alns_solver.build()
            solution = alns_solver.resume(tour=solution, iterations = 15000)
        else:
            raise NotImplementedError

        end = time.time()
        #print("solution (after): ", solution, feasible)
        #precedence_check, tw_check, capacity_check, error, violatedLoc, locTime = check1PDPTW(solution, instance, return_now=False)
        #print(precedence_check, tw_check, capacity_check, error, violatedLoc, locTime)

        cost = cost_func(solution, W, E, L, beta=self.args.beta)
        status = self.get_status(instance, solution)
        return solution, cost, end-start, status

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args = dotdict({
        'dataset_name': '1PDPTW_generated_d11_i100000_tmin100_tmax300_sd2022',
        'emb_dim'              : 20,
        'emb_iter_T'           : 1,
        'num_episodes'         : 30001,
        'batch_size'           : 32,
        'lr'                   : 5e-3,
        'lr_decay_rate'        : 1. - 2e-5,
        'beta'                 : 1,
        'repair'               : 'mip_cplex',
        'repair_strategy'      : 0
        'beta_alns'            : 10,
        'epsilon'              : 0.05,
        'degree_of_destruction': 0.6,

        'seed'                 : 6
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

    agent = RLAgent_repair(args, model_dir=model_dir, device=device)
    #agent = ALNSAgent(args)
    instance = read1PDPTW('data/1PDPTW_generated_d11_i3000_tmin100_tmax300_sd2022_test/INSTANCES/generated-3.txt')
    solution = agent.solve(instance)
    print(solution)