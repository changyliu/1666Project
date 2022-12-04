import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import os
from collections import namedtuple

from utils import get_static_state

import config as c
config = c.config()

def init_model(*args, model_name, fname=None, device=torch.device('cpu')):
    #Create a new model. If fname is defined, load the model from the specified file.
    args = args[0]

    Q_net = QNet(args.emb_dim, T=args.emb_iter_T, device=device).to(device)
    optimizer = optim.Adam(Q_net.parameters(), lr=args.lr)
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay_rate)
    
    if fname is not None:
        checkpoint = torch.load(fname, map_location=torch.device('cpu'))
        Q_net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    
    Q_func = QLearning(Q_net, optimizer, lr_scheduler, model_name=model_name, device=device)
    return Q_func, Q_net, optimizer, lr_scheduler

class QNet(nn.Module):
    """
    The neural net that will parameterize the function Q(s, a)
    
    The input is the state (containing the graph and visited nodes),
    and the output is a vector of size N containing Q(s, a) for each of the N actions a.
    """    
    
    def __init__(self, emb_dim, node_dim=9, T=4, device=torch.device('cpu')):
        """ 
        We use 9 dimensions for representing the nodes' states:
        (binary) whether the node has been visited
        (binary) whether the node is the first of the visited sequence
        (binary) whether the node is the last of the visited sequence
        (int) x,
        (int) y,
        (int) entering time,
        (int) leaving time,
        (int) demand,
        (float) distance to the depot

        emb_dim: embedding dimension p
        T: number of iterations for the graph embedding

        """
        super(QNet, self).__init__()
        self.emb_dim = emb_dim
        self.T = T
        self.device = device
        
        self.node_dim = node_dim
        
        # We can have an extra layer after theta_1 (for the sake of example to make the network deeper)
        nr_extra_layers_1 = 1
        
        # Build the learnable affine maps:
        self.theta1 = nn.Linear(self.node_dim, self.emb_dim, True)
        self.theta2 = nn.Linear(self.emb_dim, self.emb_dim, True)
        self.theta3 = nn.Linear(self.emb_dim, self.emb_dim, True)
        self.theta4 = nn.Linear(1, self.emb_dim, True)
        self.theta5 = nn.Linear(2*self.emb_dim, 1, True)
        self.theta6 = nn.Linear(self.emb_dim, self.emb_dim, True)
        self.theta7 = nn.Linear(self.emb_dim, self.emb_dim, True)
        
        self.theta1_extras = [nn.Linear(self.emb_dim, self.emb_dim, True) for _ in range(nr_extra_layers_1)]
        
    def forward(self, xv, Ws):
        # xv: The node features (batch_size, num_nodes, node_dim)
        # Ws: The graphs (batch_size, num_nodes, num_nodes)
        
        num_nodes = xv.shape[1]
        batch_size = xv.shape[0]
        
        # pre-compute 1-0 connection matrices masks (batch_size, num_nodes, num_nodes)
        conn_matrices = torch.where(Ws > 0, torch.ones_like(Ws), torch.zeros_like(Ws)).to(self.device)
        
        # Graph embedding
        # Note: we first compute s1 and s3 once, as they are not dependent on mu
        mu = torch.zeros(batch_size, num_nodes, self.emb_dim, device=self.device)
        s1 = self.theta1(xv)  # (batch_size, num_nodes, emb_dim)

        #for layer in self.theta1_extras:
        #    s1 = layer(F.relu(s1))  # we apply the extra layer
        
        s3_1 = F.relu(self.theta4(Ws.unsqueeze(3)))  # (batch_size, nr_nodes, nr_nodes, emb_dim) - each "weigth" is a p-dim vector        
        s3_2 = torch.sum(s3_1, dim=1)  # (batch_size, nr_nodes, emb_dim) - the embedding for each node
        s3 = self.theta3(s3_2)  # (batch_size, nr_nodes, emb_dim)
        
        for t in range(self.T):
            s2 = self.theta2(conn_matrices.matmul(mu))    
            mu = F.relu(s1 + s2 + s3)
            
        # we repeat the global state (summed over nodes) for each node, 
        # in order to concatenate it to local states later
        global_state = self.theta6(torch.sum(mu, dim=1, keepdim=True).repeat(1, num_nodes, 1))
        
        local_action = self.theta7(mu)  # (batch_dim, nr_nodes, emb_dim)
            
        out = F.relu(torch.cat([global_state, local_action], dim=2))
        return self.theta5(out).squeeze(dim=2)

class QLearning():
    def __init__(self, model, optimizer, lr_scheduler, device=torch.device('cpu'), model_name='sample'):
        self.model = model  # The actual QNet
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = nn.MSELoss()

        self.device = device
        self.model_name = model_name
        self.checkpoint_path = os.path.join(
                                    '.', 
                                    config['MODEL_DIR'], 
                                    model_name
                                    )
    
    def predict(self, state_tsr, W):
        # batch of 1 - only called at inference time
        with torch.no_grad():
            estimated_rewards = self.model(state_tsr.unsqueeze(0), W.unsqueeze(0))
        return estimated_rewards[0]
                
    def get_best_action(self, state_tsr, state):
        """ 
        Computes the best (greedy) action to take from a given state
        Returns a tuple containing the ID of the next node and the corresponding estimated reward

        """
        W = state.W
        estimated_rewards = self.predict(state_tsr, W)  # size (nr_nodes,)
        sorted_reward_idx = estimated_rewards.argsort(descending=True)
                
        for idx in sorted_reward_idx.tolist():
            if self.is_valid_action(idx, state):
                return idx, estimated_rewards[idx].item()

    def batch_update(self, states_tsrs, Ws, actions, targets):
        """
        Take a gradient step using the loss computed on a batch of (states, Ws, actions, targets)
        
            states_tsrs: list of (single) state tensors
            Ws: list of W tensors
            actions: list of actions taken
            targets: list of targets (resulting estimated rewards after taking the actions)
        """        
        Ws_tsr = torch.stack(Ws).to(self.device)
        xv = torch.stack(states_tsrs).to(self.device)
        self.optimizer.zero_grad()
        
        # the rewards estimated by Q for the given actions
        estimated_rewards = self.model(xv, Ws_tsr)[range(len(actions)), actions]
        
        loss = self.loss_fn(estimated_rewards, torch.tensor(targets, device=self.device))
        loss_val = loss.item()
        
        loss.backward()
        self.optimizer.step()        
        self.lr_scheduler.step()
        
        return loss_val

    def get_next_neighbor_random(self, state):
        solution, W = state.partial_solution, state.W
        
        if len(solution) == 0:
            return random.choice(range(W.shape[0]))

        neighbors = W[solution[-1]].nonzero()
        candidates = []
        for idx in neighbors:
            if self.is_valid_action(idx, state):
                candidates.append(idx)

        if len(candidates) == 0:
            return None
        return random.choice(candidates).item()

    def is_valid_action(self, idx, state):
        """
        Check if going to node 'idx' is valid as a next action.

        Going to node 'idx' is valid if (1)-(4) are satisfied:
        1) there's a path from the last node in the partial solution to this idx
        2) 'idx' is not in the partial solution
        3) if it is a delivery node, pick up is done
        4) load will not exceed the capacity.

        idx (int)          : a node
        state (namedTuple) : state
        
        """
        solution = state.partial_solution
        pickup = state.pickup
        #print(idx)
        #print("{} (load) + {} (demand) <= {} (cap)".format(state.load, state.demands[idx], state.capacity))

        if (len(solution) == 0 or state.W[solution[-1], idx] > 0) and \
            idx not in solution and \
            (pickup[idx] == 0 or (pickup[idx] - 1) in solution) and \
            state.load + state.demands[idx] <= state.capacity:
                return True
        return False

    def checkpoint_model(self, loss, episode, avg_cost):
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        
        fname = os.path.join(self.checkpoint_path, 'ep_{}'.format(episode))
        fname += '_cost_{}'.format(int(avg_cost))
        fname += '.tar'
        
        torch.save({
            'episode': episode,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'loss': loss,
            'avg_cost': avg_cost
        }, fname)

def state2tens(state, device):
    """
    Creates a Pytorch tensor representing the history of visited nodes, from a (single) state tuple.
        
    Returns a (Nx9) tensor, where for each node we store whether this node is in the sequence,
    whether it is first or last, and its (x,y) coordinates.

    State space (9D):
        (binary) whether the node has been visited,
        (binary) Is it the first node?,
        (binary) Is it the last node in the partial solution?,
        (int) x,
        (int) y,
        (int) entering time,
        (int) leaving time,
        (int) demand,
        (float) distance to the depot

    """
    solution = state.partial_solution
    sol_last_node = state.partial_solution[-1] if len(state.partial_solution) > 0 else -1
    sol_first_node = state.partial_solution[0] if len(state.partial_solution) > 0 else -1
    coords = state.coords
    entering_times = state.entering_times
    leaving_times = state.leaving_times
    demands = state.demands
    W = state.W
    nr_nodes = coords.shape[0]

    xv = [[(1 if i in set(solution) else 0),
           (1 if i == sol_first_node else 0),
           (1 if i == sol_last_node else 0),
           coords[i,0],
           coords[i,1],
           entering_times[i],
           leaving_times[i],
           demands[i],
           W[i, 0],
          ] for i in range(nr_nodes)]
    
    return torch.tensor(xv, dtype=torch.float32, requires_grad=False, device=device)

def is_state_final(state):
    return len(set(state.partial_solution)) == state.W.shape[0]

def gen_solution(Q_func, instance, device=torch.device('cpu')):
    coords, capacity, demands, pickup, delivery, W_np, E, L = get_static_state(instance)
    W = torch.tensor(W_np, dtype=torch.float32, requires_grad=False, device=device)
    
    load = 0
    solution = [0]
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
    current_state_tsr = state2tens(current_state, device)
        
    while not is_state_final(current_state):
        next_node, est_reward = Q_func.get_best_action(current_state_tsr, 
                                                    current_state)
                        
        solution = solution + [next_node]
        if load > current_state.capacity:
            raise ValueError
        load += demands[next_node]

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
        current_state_tsr = state2tens(current_state, device)
    
    return solution, coords, W, E, L

State = namedtuple('State', 
                    ('W', 
                     'coords', 
                     'partial_solution', 
                     'entering_times', 
                     'leaving_times',
                     'capacity',
                     'load',
                     'demands',
                     'pickup',
                     'delivery'
                     )
                    )