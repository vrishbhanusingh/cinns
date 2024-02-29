# from CINNS_20_7_2023 import CINNS
from diagnn import *
import os
import glob
import torch
import shutil
import random
import datetime
import itertools
import numpy as np
from gym import Env
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from prettytable import PrettyTable
from gym.spaces import MultiDiscrete, Box, Tuple, Discrete

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

import warnings

# Enable runtime warnings
warnings.simplefilter("always", RuntimeWarning)

IN_COLAB = torch.cuda.is_available()

path = os.getcwd()

def print_table(lrate, betas, solver_optimizer, s_v):
    pt = PrettyTable()
    pt.field_names = ['Action Name','Action Taken']
    pt.add_row(['Learning_rate',"{:e}".format(lrate)])
    pt.add_row(['Betas', betas])
    pt.add_row(['Optimizer', solver_optimizer])
    pt.add_row(['Scales', s_v])
    print(pt)

class CINNS_RL(Env):
    
    def __init__(self, n_curriculum_options = 10, data_path = "Data/Curve_sofT_Case3of5.csv"):
        # self.c = CINNS("Data/Curve_sofT_case3of5.csv", 7)
        self.data_path = data_path
        self.c = DiagNN(data_path)
        self.num_epochs_per_update = 30
        self.reward = 0.0
        self.step_count = 0
        self.gamma = 0.99
        self.curriculum = 1
        self.curriculum_options = self.generate_curriculum_options(n_curriculum_options)
        # # Possible options
        
        # self.learning_rate_options = np.concatenate((np.linspace(1e-4, 9e-4, 9), np.linspace(1e-3, 1e-2, 10)), axis=0)  

        # self.beta_values = [0.8, 0.88, 0.888, 0.9, 0.99, 0.999]
        # self.betas_options = list(itertools.combinations(self.beta_values, 2)) 


        # self.scales = np.linspace(0,1.0,11)

        # # Action Space
        
        # self.n_learning_rates = self.learning_rate_options.shape[0]
        # self.n_betas = len(self.betas_options)
        # self.n_scales = self.scales.shape[0]
        
        # self.action_space = MultiDiscrete([self.n_learning_rates, self.n_betas, 2] + [self.n_scales]*7)
        
        # Action Space
        # self.low_lr = 1e-4
        # self.high_lr = 1e-2
        # self.low_beta = 0.8
        # self.high_beta = 0.999
        # self.low_scale = 0.0
        # self.high_scale = 1.0

        self.action_space = Discrete(10)
        # Observation Space
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(7, 16, 41), dtype=np.float32)
        
        # self.c.opts.scale=2 #just for the first iteration, train without scaling
        self.c.update_generator(self.curriculum)
        self.c.solver.fit(max_epochs=self.num_epochs_per_update, tqdm_file=None)
        # self.c.opts.scale=1 #optimize the scaling after 1st iteration
        self.state = self.c.get_residuals()
        self.loss = self.c.get_loss()
        self.episode_returns = 0.0

        # self.n_actions_per_dimension = [self.n_learning_rates, self.n_betas, 2, *([self.n_scales] * 7)]
    def generate_curriculum_options(self,n):
        x = np.linspace(0, 1, n)
        return (9*(1 - (1-x)**2)+1)/10
    def map_action_to_curiculum(self, action):
        possible_actions = list(range(10))
        possible_curriculum_options = self.curriculum_options
        actions2curriculum = {num: val for num, val in zip(possible_actions, possible_curriculum_options)}
        curriculum_value = actions2curriculum[action]
                                               
        return curriculum_value 
    
    def get_reward(self, new_loss, old_loss):
        
        step_penalty = self.gamma ** (self.step_count - 1)
                                               
        # scales_penalty = len(s_v) - np.count_nonzero(s_v)
        
        
        with warnings.catch_warnings(record=True) as warning_list:
            
            log_reward = np.log10(np.abs(old_loss - new_loss))*10
            simple_reward = 1/self.loss 
        
        for warning in warning_list:
            if issubclass(warning.category, RuntimeWarning):
                print(f"Warning message: {warning.message}")
                print(reward)
                print(f"Variable 1: {old_loss}")
                print(f"Variable 2: {new_loss}")

        # if old_loss - new_loss < 0:

        #     if np.abs(old_loss - new_loss) < 1:
        #         reward = log_reward - scales_penalty
        #     else:
        #         reward = -log_reward - scales_penalty
        # else:
                                  
        #     if np.abs(old_loss - new_loss) < 1:
        #         reward = -log_reward - scales_penalty
        #     else:
        #         reward = log_reward - scales_penalty
        reward = log_reward                 
        return step_penalty * reward
        # return step_penalty * simple_reward
    
    def step(self, action, show_actions = False):
        # if self.step_count > (15000//self.num_epochs_per_update):
        #     reward = self.reward
        #     self.state = self.reset()
        #     done = True
        #     return self.state, reward ,done 
        done = False
        action = action.item()
        curriculum_action = self.map_action_to_curiculum(action)

        self.c.update_generator(curriculum_action)
        
        # if self.step_count % 30 == 0 and self.step_count !=0:
        #     if self.curriculum == 1.0:
        #         pass
        #     else:
        #         self.curriculum = round(self.curriculum + 0.1, 1)
        #         self.c.update_generator(self.curriculum)

        # if show_actions:
        #     print_table(lrate, betas, solver_optimizer, s_v)
        
        # if solver_optimizer == 0:
        #     self.c.lbfgs = torch.optim.LBFGS(set([p for net in self.c.nets + [self.c.V] for p in net.parameters()]), \
        #                 lr=3e-3)
        #     self.c.solver.optimizer = self.c.lbfgs
        # else:

        #     self.c.adam = torch.optim.Adam(set([p for net in self.c.nets + [self.c.V] for p in net.parameters()]), \
        #                 lr=lrate,  betas=betas)
        #     self.c.solver.optimizer = self.c.adam
        
        self.c.solver.fit(max_epochs=self.num_epochs_per_update, tqdm_file=None)
        
        self.state = self.c.get_residuals(show_actions)
        prev_loss = self.loss
                                               
        self.loss = self.c.get_loss()
            
        self.reward = self.reward + self.get_reward(self.loss, prev_loss)  

        self.step_count = self.step_count + 1
        
        self.episode_returns += self.reward
        return self.state, self.reward, done, {}, {}
        
    def render(self):
        
        self.c.plot_loss()
        self.c.plot_residuals(self.state)
        self.c.plot_result()
        self.c.plot_best_result()
     
    def reset(self):
        
        done = False
        self.episode_returns = 0.0
        self.reward = 0.0
        self.curriculum = 0.1
        self.step_count = 0
        self.c = DiagNN(self.data_path)
        # self.c.opts.scale=2 #just for the first iteration, train without scaling
        self.c.update_generator(self.curriculum)
        self.c.solver.fit(max_epochs=self.num_epochs_per_update, tqdm_file=None)
        self.state = self.c.get_residuals()
        self.loss = self.c.get_loss()
        # self.c.opts.scale=1

        ## **changed return value**
        return self.state, self.reward, done, {}, {}
    
