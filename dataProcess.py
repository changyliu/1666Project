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

def txt_to_pdptw_all(dir):
    ins_dir = join(dir, 'INSTANCES')
    assert os.path.exists(ins_dir)
    for file in os.listdir(ins_dir):
        name, ext = file.split('.')
        if ext == 'txt':
            txt_to_pdptw(join(ins_dir, file))

def txt_to_pdptw(file_path):
    """
    Conver an instance txt file to a pdptw file so LKH can read the file.

    """
    print(file_path)

    f = open(file_path, "r")

    name = f.readline().split()[-1]
    f.readline() # skip TYPE
    numLocation = int(f.readline().split()[-1])
    numVehicle = int(f.readline().split()[-1])
    capacity = int(f.readline().split()[-1])
    egdeWeightType = f.readline().split()[-1]

    f.readline() #skip title line NODE_COORD_SECTION
    coord = []
    for i in range(numLocation):
        line = f.readline().split()
        coord.append((int(line[1]), int(line[2])))

    pd = []
    f.readline() #skip title line PICKUP_AND_DELIVERY_SECTION
    for i in range(numLocation):
        line = f.readline().split()
        pd.append(line)

    lines = []
    lines.append(f'NAME : {name}')
    lines.append('TYPE : PDPTW')
    lines.append(f'DIMENSION : {numLocation}')
    lines.append(f'VEHICLES : {numVehicle}')
    lines.append(f'CAPACITY : {capacity}')
    lines.append(f'EDGE_WEIGHT_TYPE : {egdeWeightType}')

    lines.append('NODE_COORD_SECTION')
    for loc in range(numLocation):
        lines.append(f'{loc+1} {coord[loc][0]} {coord[loc][1]}')

    lines.append('PICKUP_AND_DELIVERY_SECTION')
    for loc in range(numLocation):
        lines.append(f'{pd[loc][0]} {pd[loc][1]} {pd[loc][2]} {pd[loc][3]} {pd[loc][4]} {pd[loc][5]} {pd[loc][6]}')

    lines.append('DEPOT_SECTION')
    lines.append('1')
    lines.append('-1')
    lines.append('EOF')

    seg = re.split('[/\\\]', file_path)
    save_dir = ''
    for _s in seg[:-2]:
        save_dir += _s + '/'
    save_dir += 'INSTANCES_LKH'
    os.makedirs(save_dir, exist_ok=True)

    f = open(join(save_dir, f'{name}.pdptw'), "w") # write instance file
    for l in lines:
        f.write(l + '\n')
    f.close()

def make_runAll_PDPTW_ours(dir, max_trials=1000, runs=1):
    """
    Make a shell script that runs LKH for all the instances in the directory.

    """
    ins_dir = join(dir, 'INSTANCES_LKH')
    assert os.path.exists(ins_dir)

    lines = []
    for file in os.listdir(ins_dir):
        name = file.split('.')[0]
        lines.append(f'./run_PDPTW {name} {max_trials} {runs}')

    f = open(join(dir, 'runAll_PDPTW_ours'), "w") # write instance file
    for l in lines:
        f.write(l + '\n')
    f.close()

def readLKHTour(file_path):
    """
    Read a LKH tour ('{instance-name}-tour.txt') and return the tour.

    """

    f = open(file_path, "r")
    tour = f.readline().split()
    tour = [int(t) for t in tour]
    return tour

def get_results_LKH(dir):
    """
    Read all the LKH tours in $dir and check the solution feasibility.

    """
    ins_dir = join(dir, 'INSTANCES_LKH')
    assert os.path.exists(ins_dir)

    from solnCheck import check1PDPTW_all

    num_feasible = 0
    for file in os.listdir(ins_dir):
        line = file.split('.')
        name = line[0]
        ext = line[1]
        if ext == 'txt' and 'tour' in name:
            solution = readLKHTour(join(ins_dir, file))
            if len(solution) > 0:
                instance = read1PDPTW(join(dir, 'INSTANCES', f'{name[:-5]}.txt'))
                p_check, tw_check, c_check, error, error_dict, _, _ = check1PDPTW_all(solution, instance, return_now=False)
                print(f'{name}: p{int(p_check)} tw{int(tw_check)} c{int(c_check)} num_error={len(error)}')

                if len(error) == 0:
                    num_feasible += 1
    print("Num feasible: ", num_feasible)

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
