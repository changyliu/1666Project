from abc import ABC, abstractmethod

import os
import torch

from dataProcess import read1PDPTW
from solnRepair import solnRepair
from solnCheck import check1PDPTW
from models import init_model, gen_solution
from utils import get_best_model, dotdict, float_to_str, cost_func

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

    def get_status(self, instance, solution):
        # feasibility check
        _soln = [x+1 for x in solution]
        p_check, tw_check, c_check, error, _, _ = check1PDPTW(_soln, instance, return_now=False)
        if len(error) == 0:
            return 'feasible'
        else:
            return 'infeasible'

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

        solution = [x+1 for x in solution]
        #print("solution (before): ", solution)
        #precedence_check, tw_check, capacity_check, error, violatedLoc, locTime = check1PDPTW(solution, instance, return_now=False)
        #print(precedence_check, tw_check, capacity_check, error, violatedLoc, locTime)

        solution, numIter, timeSpent, feasible = solnRepair(solution, instance, 5000, 600)
        end = time.time()
        solution = [x-1 for x in solution]
        #print("solution (after): ", solution, feasible)
        #precedence_check, tw_check, capacity_check, error, violatedLoc, locTime = check1PDPTW(solution, instance, return_now=False)
        #print(precedence_check, tw_check, capacity_check, error, violatedLoc, locTime)

        cost = cost_func(solution, W, E, L, beta=self.args.beta)
        status = self.get_status(instance, solution)
        return solution, cost, end-start, status

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args = dotdict({
        'dataset_name': '1PDPTW_generated',
        'emb_dim'      : 20,
        'emb_iter_T'   : 1,
        'num_episodes' : 20001,
        'batch_size'   : 32,
        'lr'           : 5e-3,
        'lr_decay_rate': 1. - 2e-5,
        'beta'         : 1,
        'seed'         : 6
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
    instance = read1PDPTW('data/1PDPTW_generated_test/INSTANCES/generated-1004.txt')
    solution = agent.solve(instance)
    print(solution)