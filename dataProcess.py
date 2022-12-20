# Read and Process Data

from os.path import isfile, join
import os
from tqdm import tqdm
import csv

def to_csv(output_dict, path):
    """
    output_dict (dict) : dictionary object that stores experiment outputs.
    path (str)         : path to save the csv file.

    """
    contents_names = ['instance', 'solution', 'cost']
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

def read1PDPTW_tour(file_path):
    f = open(file_path, "r")

    cost = f.readline().split()[-1]
    line = f.readline().split()
    tour = [int(x) for x in line]
    return cost, tour

def read1PDPTW(file_path):
    f = open(file_path, "r")

    instance = {}
    instance['name'] = f.readline().split()[-1]
    instance['type'] = f.readline().split()[-1]
    instance['numLocation'] = int(f.readline().split()[-1])
    instance['numVehicle'] = int(f.readline().split()[-1])
    instance['capacity'] = int(f.readline().split()[-1])
    instance['egdeWeightType'] = f.readline().split()[-1]

    f.readline() #skip title line NODE_COORD_SECTION
    coord = []
    for i in range(instance['numLocation']):
        line = f.readline().split()
        coord.append((int(line[1]), int(line[2])))
    instance['coordinates'] = coord

    demand = []
    tw = []
    serviceTime = []
    pickup = []
    delivery = []
    f.readline() #skip title line PICKUP_AND_DELIVERY_SECTION
    for i in range(instance['numLocation']):
        line = f.readline().split()
        demand.append(int(line[1]))
        tw.append((int(line[2]), int(line[3])))
        serviceTime.append(int(line[4]))

        p = int(line[5])
        #if p > 0:
        #    assert p > 1, \
        #        "p != 1 since node idx starts from 1 and node 1 is depot"
        #    p -= 1
        pickup.append(p)

        d = int(line[6])
        #if d > 0:
        #    assert d > 1, \
        #        "d != 1 since node idx starts from 1 and node 1 is depot"
        #    d -= 1
        delivery.append(d)

    instance['demand'] = demand
    instance['tw'] = tw
    instance['serviceTime'] = serviceTime
    instance['pickup'] = pickup
    instance['delivery'] = delivery

    return instance


if __name__ == "__main__":
    # instance = read1PDPTW('data/1PDPTW_generated/INSTANCES/generated-11-0.txt')
    # print(instance['name'])

    dataset_name = '1PDPTW_generated_d51_i200_tmin300_tmax500_sd2022_test'
    dataset_path = 'data/1PDPTW_generated_d51_i200_tmin300_tmax500_sd2022_test/TOURS'

    result_dir = os.path.join('.', 'results', 'experiment', dataset_name)
    os.makedirs(result_dir, exist_ok=True)
    result_filename = '{}_rp{}_{}'.format(
                'feasible_tour',
                'none',
                dataset_name
                )
    filenames = []
    solutions = []
    costs = []
    for file in tqdm(os.listdir(dataset_path)):
        if not file.startswith('.'):
            cost, tour = read1PDPTW_tour(os.path.join(dataset_path, file))
            filenames.append(file)
            solutions.append(tour)
            costs.append(cost)

    output_dict = {}
    for i, (file, soln, cost) in enumerate(zip(filenames, solutions, costs)):
        data = {'instance'  : file, 
                'solution'  : [int(x) for x in soln],
                'cost'      : cost,
                }
        output_dict[i] = data

    csv_path = os.path.join(result_dir, '{}.csv'.format(result_filename))
    to_csv(output_dict, csv_path)
