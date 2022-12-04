import argparse
from cmath import log
import os
import torch
import json
import csv

from dataProcess import read1PDPTW
from exactModelsCplex import solve1PDPTW_MIP_CPLEX
from Agent import RLAgent, RLAgent_repair, ALNSAgent
from utils import float_to_str, dotdict
from multiprocessing import Process

import config as c
config = c.config()

class Experiment():
    def __init__(self, args, method, test_dataset):
        self.args = args
        self.method = method
        self.test_dataset = test_dataset

        if method in ['rl', 'rl_repair']:
            self.model_name = '{}_ed{}_ne{}_bs{}_lr{}_bt{}_sd{}'.format(
                            args.dataset_name,
                            args.emb_dim,
                            args.num_episodes,
                            args.batch_size,
                            float_to_str(args.lr),
                            args.beta,
                            args.seed
                            )
            self.model_dir = os.path.join('.', config['MODEL_DIR'], self.model_name)
            if method == 'rl':
                self.agent = RLAgent(
                                args, 
                                model_dir=self.model_dir, 
                                device=args.device
                                )
            elif method == 'rl_repair':
                self.agent = RLAgent_repair(
                                args, 
                                model_dir=self.model_dir, 
                                device=args.device
                                )
            else:
                raise NotImplementedError
        elif method == 'alns':
            self.agent = ALNSAgent(args)
        else:
            raise NotImplementedError
    
    def run(self, instance):
        """
        Solve one instance.
        
        """

        if self.method == 'mip':
            # Note that MIP cost considers only tour length, not time windows
            # If time window is violated, it returns infeasible
            solution, cost, solve_time, status = solve1PDPTW_MIP_CPLEX(instance, logtoconsole=False)
            # if status == 2:
            #     status = 'optimal'
            # elif status == 9:
            #     status = 'feasible'
            # elif status == 3:
            #     status = 'infeasible'
            # else:
            #     raise ValueError
        elif self.method in ['rl', 'rl_repair', 'alns']:
            solution, cost, solve_time, status = self.agent.solve(instance)
        else:
            raise NotImplementedError
        return solution, cost, solve_time, status

    def run_all(self):
        filenames = []
        solutions = []
        costs = []
        solve_times = []
        status_all = []
        dataset_path = os.path.join('.', config['DATA_DIR'], self.test_dataset, 'INSTANCES')
        for file in os.listdir(dataset_path):
            print(file)
            instance = read1PDPTW(os.path.join(dataset_path, file))
            soln, cost, solve_time, status = self.run(instance)

            filenames.append(file)
            solutions.append(soln)
            solve_times.append(solve_time)
            status_all.append(status)
            costs.append(cost)
        
        feasible_num = sum([1 for s in status_all if s in ['feasible', 'optimal']])
        feasible_rate = feasible_num / len(status_all)

        output_dict = {}
        for i, (file, soln, t, status, cost) in enumerate(zip(filenames, solutions, solve_times, status_all, costs)):
            data = {'instance'  : file, 
                    'solution'  : [int(x) for x in soln],
                    'cost'      : cost,
                    'solve_time': t,
                    'status'    : status,
                    }
            output_dict[i] = data

        # Save the results to json and csv
        result_dir = os.path.join('.', config['RESULT_DIR'], 'experiment', self.test_dataset)
        os.makedirs(result_dir, exist_ok=True)
        result_filename = '{}_rp{}_trd{}_ed{}_ne{}_bs{}_lr{}_bt{}_dst{}_sd{}'.format(
                    self.method,
                    self.args.repair,
                    self.args.dataset_name,
                    self.args.emb_dim,
                    self.args.num_episodes,
                    self.args.batch_size,
                    float_to_str(self.args.lr),
                    self.args.beta,
                    float_to_str(self.args.degree_of_destruction),
                    self.args.seed
                    )
        json_path = os.path.join(result_dir, '{}.json'.format(result_filename))
        self.to_json(output_dict, json_path, feasible_rate=feasible_rate)

        csv_path = os.path.join(result_dir, '{}.csv'.format(result_filename))
        self.to_csv(output_dict, csv_path)

        return solutions, costs, solve_times, status_all

    def to_csv(self, output_dict, path):
        """
        output_dict (dict) : dictionary object that stores experiment outputs.
        path (str)         : path to save the csv file.
        
        """
        contents_names = ['instance', 'solution', 'cost', 'solve_time', 'status']
        output = []
        for (key, val) in output_dict.items():
            tmp = []
            try:
                for content in contents_names:
                    tmp.append(val[content])
                output.append(tmp)
            except:
                print("broken json file", val)
                pass

        with open(path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(contents_names)
            writer.writerows(output)

    def to_json(self, output_dict, path, feasible_rate=0.0):
        """
        output_dict (dict) : dictionary object that stores experiment outputs.
        path (str)         : path to save the json file.
        
        """

        json_output = dict()
        json_output['method'] = self.method
        json_output['test_dataset'] = self.test_dataset
        json_output['train_dataset'] = self.args.dataset_name
        json_output['emb_dim'] = self.args.emb_dim
        json_output['num_episodes'] = self.args.num_episodes
        json_output['barch_size'] = self.args.batch_size
        json_output['lr'] = self.args.lr
        json_output['beta'] = self.args.beta
        json_output['seed'] = self.args.seed
        json_output['feasible_rate'] = feasible_rate
        json_output['repair'] = self.args.repair
        json_output['beta_alns'] = self.args.beta_alns
        json_output['epsilon'] = self.args.beta_alns
        json_output['degree_of_destruction'] = self.args.degree_of_destruction

        output = []
        for (key, val) in output_dict.items():
            output.append(val)

        json_output['results'] = output
        with open(path, mode='wt', encoding='utf-8') as file:
            json.dump(json_output, file, ensure_ascii=False, sort_keys=True, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--method", type=str, default='mip')
    parser.add_argument("--test-dataset", type=str, default='1PDPTW_generated_d21_i1000_tmin300_tmax500_sd2022_test')

    # RL args
    parser.add_argument("--dataset-name", type=str, default="1PDPTW_generated_d21_i100000_tmin300_tmax500_sd2022")
    parser.add_argument("--emb-dim", type=int, default=20) # Embedding dimension D
    parser.add_argument("--num-episodes", type=int, default=30001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--beta", type=int, default=1) # penalty factor for tw
    parser.add_argument("--seed", type=int, default=6)

    args, remaining = parser.parse_known_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    agent_args = dotdict({
        'dataset_name'          : args.dataset_name,
        'emb_dim'               : args.emb_dim,
        'emb_iter_T'            : 1,
        'num_episodes'          : args.num_episodes,
        'batch_size'            : args.batch_size,
        'lr'                    : args.lr,
        'lr_decay_rate'         : 1. - 2e-5,
        'beta'                  : args.beta,

        'repair'                : 'alns',

        'beta_alns'             : 10,
        'epsilon'               : 0.05,
        'degree_of_destruction' : 0.6,

        'seed'         : args.seed,
        'device'       : device
    })

    experiment = Experiment(agent_args, args.method, args.test_dataset)
    #instance = read1PDPTW('data/1PDPTW_generated_test/INSTANCES/generated-16.txt')
    solutions = experiment.run_all()
    