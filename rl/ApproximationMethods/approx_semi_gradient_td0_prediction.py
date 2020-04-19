#!/usr/bin/env python
# coding: utf-8

# In[5]:


import sys


# In[6]:


sys.path.append("C:\\Users\\mohan\\Documents\\GitHub\\Reinforcement-Learning\\rl\\DynamicProgramming")


# In[7]:


sys.path.append("C:\\Users\\mohan\\Documents\\GitHub\\Reinforcement-Learning\\rl\\MonteCarlo")


# In[8]:


sys.path.append("C:\\Users\\mohan\\Documents\\GitHub\\Reinforcement-Learning\\rl\\TemporalDifferenceLearning")


# In[9]:


import numpy as np
from matplotlib import pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy
from td0_prediction import play_game, SMALL_ENOUGH, ALPHA, GAMMA, ALL_POSSIBLE_ACTION


# In[22]:


class Model:
    def __init__(self):
        #initializing theta vector randomly.
        self.theta = np.random.randn(4)/2
    
    def state_s_to_feature_x(self, s):
        #where x is [row, column, row*column, 1] - for bias term
        return np.array([s[0] - 1, s[1] - 1.5, s[0]*s[1]-3, 1])
    
    def predict(self, s):
        #Returns the predicted value of state given a state
        x = self.state_s_to_feature_x(s)
        return self.theta.dot(x)
    
    def grad(self, s):
        #This function is not needed
        return self.state_s_to_feature_x(s)


# In[25]:


def main():
    #using the standard grid again (0 for every step) so that we can compare to iterative policy evaluation.
    grid = standard_grid()
    
    #print rewards
    print ("rewards")
    print_values(grid.rewards, grid)
    
        #Policy - For a given state what is the action we would take.
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (1, 2): 'R',
        (2, 1): 'R',
        (2, 2): 'R',
        (2, 3): 'U'
    }
    
    model = Model()
    deltas = []
    k = 1.0
    for it in range(20000):
        if it % 10 == 0:
            k += 0.01
        alpha = ALPHA/k
        #generate an episode using given policy
        biggest_change = 0
        states_and_rewards = play_game(grid, policy)
        
        for t in range(len(states_and_rewards) - 1):
            s, _ = states_and_rewards[t]
            s2, r = states_and_rewards[t + 1]
            
            old_theta = model.theta.copy()
            if grid.is_terminal(s2):
                target = r
            else:
                target = r + GAMMA * model.predict(s2)
                
            model.theta += alpha * (target - model.predict(s)) * model.grad(s) 
            biggest_change = max(biggest_change, np.abs(old_theta - model.theta).sum())
        deltas.append(biggest_change)
    
    #obtain predicted values
    V = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            V[s] = model.predict(s)
        else:
            V[s] = 0
    print("Values: ")
    print_values(V, grid)

    print("Policy: ")
    print_policy(policy, grid)


# In[26]:


if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




