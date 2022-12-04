import torch
import numpy as np
import random
from collections import namedtuple

import os
import json
import argparse

from models import init_model, state2tens, is_state_final, gen_solution, State
from memory import Memory
from utils import get_static_state, total_distance, cost_func, float_to_str, get_best_model
from plot import plot_loss, plot_path_length, plot_cost, plot_solution

from dataProcess import read1PDPTW

import config as c
config = c.config()

def train(*args):
    args = args[0]

    # seed everything for reproducible results first:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    instance_dir = os.path.join(
                        '.', 
                        config["DATA_DIR"], 
                        args.dataset_name, 
                        'INSTANCES'
                        )
    num_ins = len(os.listdir(instance_dir))

    # keep track of median cost for model checkpointing
    current_min_med_cost = float('inf')

    model_name = '{}_ed{}_ne{}_bs{}_lr{}_bt{}_sd{}'.format(
                    args.dataset_name,
                    args.emb_dim,
                    args.num_episodes,
                    args.batch_size,
                    float_to_str(args.lr),
                    args.beta,
                    args.seed
                    )
    plot_dir = os.path.join('.', config['PLOT_DIR'], model_name)
    model_dir = os.path.join('.', config['MODEL_DIR'], model_name)
    result_dir = os.path.join('.', config['RESULT_DIR'], 'train')
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    memory = Memory(args.mem_capacity)

    # Note: we store state tensors in experience to compute these tensors only once later on
    Experience = namedtuple(
                    'Experience', 
                    (
                        'state', 
                        'state_tsr', 
                        'action', 
                        'reward', 
                        'next_state', 
                        'next_state_tsr'
                        )
                    )

    Q_func, Q_net, optimizer, lr_scheduler = init_model(args, model_name=model_name, device=device)

    losses = []
    path_lengths = []
    costs = []

    for episode in range(args.num_episodes):
        # sample a new random graph
        idx = np.random.choice(num_ins)
        instance = read1PDPTW(os.path.join(instance_dir, 'generated-{}.txt'.format(idx)))
        coords, capacity, demands, pickup, delivery, W_np, E, L = get_static_state(instance)
        W = torch.tensor(W_np, dtype=torch.float32, requires_grad=False, device=device)
        load = 0

        nr_nodes = instance['numLocation']

        # current partial solution - a list of node index
        solution = [0]
        
        # current state (tuple and tensor)
        current_state = State(W=W,
                            coords=coords,
                            partial_solution=solution,
                            entering_times=E,
                            leaving_times=L,
                            capacity=capacity,
                            load=load,
                            demands=demands,
                            pickup=pickup,
                            delivery=delivery
                            )
        current_state_tsr = state2tens(current_state, device=device)
        
        # Keep track of some variables for insertion in replay memory:
        states = [current_state]
        states_tsrs = [current_state_tsr]  # we also keep the state tensors here (for efficiency)
        rewards = []
        actions = []
        
        # current value of epsilon
        epsilon = max(args.min_epsilon, (1-args.epsilon_decay_rate)**episode)
        
        nr_explores = 0
        t = -1
        while not is_state_final(current_state):
            t += 1  # time step of this episode
            
            if epsilon >= random.random():
                # explore
                next_node = Q_func.get_next_neighbor_random(current_state)
                nr_explores += 1
            else:
                # exploit
                next_node, est_reward = Q_func.get_best_action(current_state_tsr, current_state)
                if episode % 50 == 0:
                    print('Ep {} | current sol: {} / next est reward: {}'.format(episode, solution, est_reward))
            next_solution = solution + [next_node]
            if load > current_state.capacity:
                raise ValueError
            load += demands[next_node]
            
            # reward observed for taking this step        
            #reward = -(total_distance(next_solution, W) - total_distance(solution, W))
            reward = -(cost_func(next_solution, W, current_state.entering_times, current_state.leaving_times) - \
                        cost_func(solution, W, current_state.entering_times, current_state.leaving_times))
            
            next_state = State(W=W,
                            coords=coords,
                            partial_solution=next_solution,
                            entering_times=E,
                            leaving_times=L,
                            capacity=capacity,
                            load=load,
                            demands=demands,
                            pickup=pickup,
                            delivery=delivery
                            )
            next_state_tsr = state2tens(next_state, device=device)
            
            # store rewards and states obtained along this episode:
            states.append(next_state)
            states_tsrs.append(next_state_tsr)
            rewards.append(reward)
            actions.append(next_node)
            
            # store our experience in memory, using n-step Q-learning:
            if len(solution) >= args.n_step_ql:
                memory.remember(Experience(state=states[-args.n_step_ql],
                                        state_tsr=states_tsrs[-args.n_step_ql],
                                        action=actions[-args.n_step_ql],
                                        reward=sum(rewards[-args.n_step_ql:]),
                                        next_state=next_state,
                                        next_state_tsr=next_state_tsr))
                
            if is_state_final(next_state):
                for n in range(1, args.n_step_ql):
                    memory.remember(Experience(state=states[-n],
                                            state_tsr=states_tsrs[-n], 
                                            action=actions[-n], 
                                            reward=sum(rewards[-n:]), 
                                            next_state=next_state,
                                            next_state_tsr=next_state_tsr))
            
            # update state and current solution
            current_state = next_state
            current_state_tsr = next_state_tsr
            solution = next_solution
            
            # take a gradient step
            loss = None
            if len(memory) >= args.batch_size and len(memory) >= 2000:
                experiences = memory.sample_batch(args.batch_size)
                
                batch_states_tsrs = [e.state_tsr for e in experiences]
                batch_Ws = [e.state.W for e in experiences]
                batch_actions = [e.action for e in experiences]
                batch_targets = []
                
                for i, experience in enumerate(experiences):
                    target = experience.reward
                    if not is_state_final(experience.next_state):
                        _, best_reward = Q_func.get_best_action(experience.next_state_tsr, 
                                                                experience.next_state)
                        target += args.gamma * best_reward
                    batch_targets.append(target)
                    
                # print('batch targets: {}'.format(batch_targets))
                loss = Q_func.batch_update(batch_states_tsrs, batch_Ws, batch_actions, batch_targets)
                losses.append(loss)

                # Save model when we reach a new low average cost func
                med_cost = np.median(costs[-100:])
                if (episode > 500) and (med_cost < current_min_med_cost):
                    current_min_med_cost = med_cost
                    Q_func.checkpoint_model(loss, episode, med_cost)
                    
        length = total_distance(solution, W)
        path_lengths.append(length)
        cost = cost_func(solution, W, current_state.entering_times, current_state.leaving_times)
        #cost = total_distance(solution, W)
        costs.append(cost)

        if episode % 10 == 0:#
            print('Ep %d. Loss = %.3f / med cost = %.3f / cost = %.3f / length = %.4f / epsilon = %.4f / lr = %.4f' % (
                episode, (-1 if loss is None else loss), np.median(costs[-50:]), cost, length, epsilon,
                Q_func.optimizer.param_groups[0]['lr']))

            if len(memory) >= args.batch_size and len(memory) >= 2000:
                plot_loss(losses, plot_dir, plot_min=10**4, plot_max=10**5)
                plot_path_length(path_lengths, plot_dir)
                plot_cost(costs, plot_dir)

    json_output = dict()
    json_output['losses'] = losses
    json_output['path_lengths'] = path_lengths
    json_output['costs'] = costs
    with open(os.path.join(result_dir, '{}.json'.format(model_name)), mode='wt', encoding='utf-8') as file:
        json.dump(json_output, file, ensure_ascii=False, sort_keys=True, indent=2)

    if len(os.listdir(model_dir)) > 0:
        best_fname, _ = get_best_model(model_dir)

        #Load checkpoint
        Q_func, _, _, _ = init_model(
                                    args, 
                                    model_name=model_name,
                                    fname=os.path.join(model_dir, best_fname), 
                                    device=device
                                    )
        
        #Generate example solutions
        for _ in range(10):
            idx = np.random.choice(num_ins)
            instance = read1PDPTW(os.path.join(instance_dir, 'generated-{}.txt'.format(idx)))

            solution, coords, W, E, L = gen_solution(Q_func, instance, device=device)

            rl_name = 'generated-{}_rl.png'.format(idx)
            rd_name = 'generated-{}_rd.png'.format(idx)
            random_solution = list(range(instance['numLocation']))

            plot_solution(plot_dir, rl_name, coords, W, solution, E, L, args.beta)
            plot_solution(plot_dir, rd_name, coords, W, random_solution, E, L, args.beta)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--emb_dim", type=int, default=20) # Embedding dimension D
    parser.add_argument("--emb_iter_T", type=int, default=1) # Number of embedding iterations T

    # Learning
    parser.add_argument("--num_episodes", type=int, default=20001)
    parser.add_argument("--mem_capacity", type=int, default=10000)
    # Number of steps (n) in n-step Q-learning to wait before computing target reward estimate
    parser.add_argument("--n_step_ql", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--lr_decay_rate", type=float, default=1. - 2e-5) # learning rate decay

    parser.add_argument("--min_epsilon", type=float, default=0.1)
    parser.add_argument("--epsilon_decay_rate", type=float, default=6e-4) # epsilon decay

    parser.add_argument("--beta", type=int, default=1) # penalty factor for tw

    parser.add_argument("--dataset_name", type=str, default="1PDPTW_generated_d11_i10000_tmin100_tmax300_sd2022")

    parser.add_argument("--seed", type=int, default=0)

    args, remaining = parser.parse_known_args()

    train(args)