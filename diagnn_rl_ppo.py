from networks import FeedForwardNN
import torch
import numpy as np
class PPO:
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space.shape
        # self.action_space = np.array([env.action_space.sample()])).shape
        # print(self.action_space, self.observation_space)
        self.init_hyperparameters()
        ###algorithm step 1###
        ### Initialize actor and critic networks ###
        self.actor = FeedForwardNN(self.observation_space, 1)
        self.critic = FeedForwardNN(self.observation_space, 1)
        
        ## Init actor optimizer ##
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr = self.lr)
        
        ## init critic optimizer ##
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr = self.lr)
        
        
        
        
    
    
    def init_hyperparameters(self):
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2
        self.lr = 0.005
        
    
    def get_action(self, obs):
        ## get logits from the actor according to the current state
        logits = self.actor(obs)
        ## Make a categorical distribution from the logits
        m = torch.distributions.Categorical(logits = logits)
        ## sample from the categorical distribution
        action = m.sample()
        ## get log probability of the sampled action
        log_prob = m.log_prob(action)
        ## return action and it's log probability
        return action, log_prob
    
    def compute_rtgs(self, batch_rews):
        ## Compute the discounted sum of rewards for each step
        batch_rtgs = []
        ## need to iterate through the rewards backwards
        
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0
            ## need to iterate through the rewards backwards to discount future rewards. 
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        ## convert the discounted rewards into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype = torch.float)
        return batch_rtgs 
    
    
    def rollout(self):
        ## batch of data ##
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = [] ## rewards to go ##
        batch_lens = [] ## episode lengths ##
        t = 0 
        
        while t < self.timesteps_per_batch:
            ## rewards this episode ##
            ep_rews = []
        
            obs,_,_,_,_ = self.env.reset()
            done = False
            
            for ep_t in range(self.max_timesteps_per_episode):
                batch_obs.append(obs)
                
                action, log_prob = self.get_action(obs)
                obs, rew, done, _ , _ = self.env.step(action)
                print('step: ', ep_t, ' reward: ',rew)
                
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                
                if done:
                    break
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
        
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens
        
    def evaluate(self, batch_obs, batch):
        V = self.critic(batch_obs).squeeze()
        ## get logits from the actor according to the current state
        logits = self.actor(batch_obs)
        ## Make a categorical distribution from the logits
        m = torch.distributions.Categorical(logits = logits)
        ## sample from the categorical distribution
        action = m.sample()
        ## get log probability of the sampled action
        log_prob = m.log_prob(action)
        return V , log_prob
            
    def learn(self, total_timesteps):
        t_so_far = 0 ## timesteps simulated so far
        
        
        ###algorithm step 2###
        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            V = self.evaluate(batch_obs)
            A_k = batch_rtgs - V.detach()
            
            A_k = (A_k - A_k.mean())/ (A_k.std() + 1e-10)
            
            for _ in range(self.n_updates_per_interation):
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
                ratios = torch.exp(curr_log_probs - batch_log_probs)
                
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                
                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = torch.nn.MSELoss()(V, batch_rtgs)
                
                
                self.actor_optim.zero_grad()
                actor_loss.backward()
                self.actor_optim.step()
                
                
                self.critic_optim.zero_grad()
                critic_loss.backward(retain_graph = True)
                self.critic_optim.step()
                
            t_so_far += np.sum(batch_lens)
                
                
                
        