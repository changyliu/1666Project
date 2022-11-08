import matplotlib.pyplot as plt
import os

from utils import moving_avg, total_distance, cost_func

import config as c
config = c.config()

def plot_graph(coords, mat):
    """
    Plot the fully connected graph.

    """

    n = len(coords)
    
    plt.scatter(coords[:,0], coords[:,1], s=[50 for _ in range(n)])
    for i in range(n):
        for j in range(n):
            if j < i:
                plt.plot([coords[i,0], coords[j,0]], [coords[i,1], coords[j,1]], 'b', alpha=0.7)
    plt.show()

def plot_solution(plot_dir, filename, coords, mat, solution, entering_times, leaving_times, beta):
    plt.figure()

    plt.scatter(coords[:,0], coords[:,1])
    n = len(coords)
    
    for idx in range(n-1):
        i, next_i = solution[idx], solution[idx+1]
        plt.plot([coords[i, 0], coords[next_i, 0]], [coords[i, 1], coords[next_i, 1]], 'k', lw=2, alpha=0.8)
    
    i, next_i = solution[-1], solution[0]
    plt.plot([coords[i, 0], coords[next_i, 0]], [coords[i, 1], coords[next_i, 1]], 'k', lw=2, alpha=0.8)
    plt.plot(coords[solution[0], 0], coords[solution[0], 1], 'x', markersize=10)

    _len = int(total_distance(solution, mat))
    cost = int(cost_func(solution, mat, entering_times, leaving_times, beta=beta))
    plt.title('model / len = {} / cost = {}'.format(_len, cost))
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()


"""
        plt.figure()
        plot_solution(coords, W, solution)
        plt.title('model / len = {} / cost = {}'.format(int(total_distance(solution, W)), int(cost_func(solution, W, E, L, beta=BETA))))
        plt.savefig('figures_v4/generated-{}_rl.png'.format(idx))
        
        # for comparison, plot a random solution
        plt.figure()
        random_solution = list(range(NR_NODES))
        plot_solution(coords, W, random_solution)
        plt.title('random / len = {} / cost = {}'.format(int(total_distance(random_solution, W)), int(cost_func(random_solution, W, E, L, beta=BETA))))
        plt.savefig('figures_v4/generated-{}_rd.png'.format(idx))
"""

def plot_loss(losses, plot_dir, plot_min=10**5, plot_max=10**6, num_avg=100):
    plt.figure(figsize=(8,5))
    plt.semilogy(moving_avg(losses, num_avg))
    plt.ylabel('loss')
    plt.xlabel('training iteration')
    plt.yticks([plot_min, plot_max])
    plt.savefig(os.path.join(plot_dir, 'loss.png'))
    plt.close()

def plot_path_length(path_lengths, plot_dir, num_avg=100):
    plt.figure(figsize=(8,5))
    plt.plot(moving_avg(path_lengths, 100))
    plt.ylabel('average length')
    plt.xlabel('episode')
    plt.savefig(os.path.join(plot_dir, 'average_length.png'))
    plt.close()

def plot_cost(costs, plot_dir, num_avg=100):
    plt.figure(figsize=(8,5))
    plt.plot(moving_avg(costs, 100))
    plt.ylabel('average cost')
    plt.xlabel('episode')
    plt.savefig(os.path.join(plot_dir, 'average_cost.png'))
    plt.close()