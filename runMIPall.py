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
from multiprocessing import Pool

import config as c
config = c.config()

def to_csv(output_dict, path):
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

def to_json(output_dict, path, dataset_name, feasible_rate=0.0):
    """
    output_dict (dict) : dictionary object that stores experiment outputs.
    path (str)         : path to save the json file.
    
    """

    json_output = dict()
    json_output['method'] = 'mip'
    json_output['test_dataset'] = dataset_name
    json_output['feasible_rate'] = feasible_rate

    output = []
    for (key, val) in output_dict.items():
        if key == 'status':
            output.append(str(val))
        else:
            output.append(val)

    json_output['results'] = output
    with open(path, mode='wt', encoding='utf-8') as file:
        json.dump(json_output, file, ensure_ascii=False, sort_keys=True, indent=2)

def task(items):
    print(items[1])
    instance = read1PDPTW(os.path.join(items[0], items[1]))
    soln, cost, solve_time, status = solve1PDPTW_MIP_CPLEX(instance, timeLimit=600)
    result = {'filename':items[1], 'soln':soln, 'cost':cost, 'solve_time':solve_time, 'status':status}
    print(result)

    return result

def runMIPall(*args):
    args = args[0]

    dataset_name = args.dataset_name
    
    filenames = []
    solutions = []
    costs = []
    solve_times = []
    status_all = []
    dataset_path = os.path.join('/home/liucha90/workplace/1666Project', 'data', dataset_name, 'INSTANCES')

    items = []
    for file in os.listdir(dataset_path):
        items.append((dataset_path,file))
    print(items)


    with Pool() as pool:
        for result in pool.map(task, items):
            filenames.append(result['filename'])
            solutions.append(result['soln'])
            solve_times.append(result['solve_time'])
            status_all.append(result['status'])
            costs.append(result['cost'])
            # print(result)

    # with multiprocessing.Pool() as pool:
	#     for result in pool.map(task, items):
    #         # print(result)
    #         filenames.append(results['filename'])
    #         solutions.append(result['soln'])
    #         solve_times.append(result['solve_time'])
    #         status_all.append(result['status'])
    #         costs.append(result['cost'])
		    
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
    result_dir = os.path.join('/home/liucha90/workplace/1666Project', 'results', 'experiment', dataset_name)
    os.makedirs(result_dir, exist_ok=True)
    result_filename = '{}_rp{}_{}'.format(
                'mip',
                'none',
                dataset_name,
                )
    # json_path = os.path.join(result_dir, '{}.json'.format(result_filename))
    # to_json(output_dict, json_path, dataset_name, feasible_rate=feasible_rate, )

    csv_path = os.path.join(result_dir, '{}.csv'.format(result_filename))
    to_csv(output_dict, csv_path)


# runMIPall('1PDPTW_generated_d21_i1000_tmin300_tmax500_sd2022_test')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--dataset_name", type=str, default='1PDPTW_generated_d21_i1000_tmin300_tmax500_sd2022_test')

    args, remaining = parser.parse_known_args()

    runMIPall(args)