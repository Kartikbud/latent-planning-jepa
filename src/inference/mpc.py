import torch
from torch import nn
import numpy as np
import heapq

"""
This is where the latent space planner is implemented.
It essentially utilizes CEM to solve the MPC problem by optimizing
for the actions that place the state after some horizon of actions closest
to the goal state. The cost is the L2 loss between the current state embedding
and the goal embedding, this is also the MSELoss in PyTorch.

zi: initial state but this will be a window of embeddings where the last embedding
    is the true state
zg: this is the goal state and is a single embedding
"""

class Planner():
    def __init__(self,
                 enc : nn.Module, # encoder model
                 pred : nn.Module, # predictor model
                 H : int = 5, # planning horizon
                 N : int = 300, # number of samples per iteration
                 K : int = 30, # number of elites per iteration
                 T : int = 30,
                 action_dim : int = 7): # number of iterations
        
        self.enc = enc
        self.pred = pred
        self.H = H
        self.N = N
        self.K = K
        self.T = T
        self.action_dim = action_dim

    @torch.no_grad()
    def CEM(self, zi, zg): 
        # zi shape: (window, embedding dim)

        mean = np.zeros((self.H, self.action_dim))
        std = np.ones((self.H, self.action_dim)) # identity matrix

        sampler = np.random.default_rng()

        for i in range(self.T):
            action_samples = []
            heapq.heapify(action_samples)
            
            for i in range(self.N): # sampling and then rolling out the actions using the model for N iterations
                actions = sampler.normal(mean, std)
                # predictor expects batch dimension so I need to unsqueeze
                actions = torch.from_numpy(actions)
                window = zi.unsqueeze(0)
                for j in range(self.H):
                    # emb = self.enc(window)                 
                    pred = self.pred(window, actions[j]).squeeze(0)[-1] # only taking the last one so shape is (192,)
                    window = torch.cat(window[1:], pred.unsqueeze(0), dim=0) # pred becomes (1, 192) and then is concatenated with the window but pops first emb
                heapq.heappush(action_samples, (nn.functional.mse_loss(window[-1], zg).item(), i, actions.numpy()))
            k_lowest_tuples = heapq.nsmallest(self.K, action_samples)
            k_lowest = np.stack([action for cost, idx, action in k_lowest_tuples]) # ndarray of shape (K, H, action_dim)

            mean = np.mean(k_lowest, axis=0)
            std = np.std(k_lowest, axis=0) + 1e-6 # the small term is added to avoid std collapse
        
        return sampler.normal(mean, std)


            


                    

