#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("C:\\Users\\mohan\\Documents\\GitHub\\Reinforcement-Learning\\rl\\DynamicProgramming")


# In[2]:


import numpy as np
from matplotlib import pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy


# In[3]:


SMALL_ENOUGH = 10e-4
ALPHA = 0.1
GAMMA = 0.9
ALL_POSSIBLE_ACTION = {'U', 'D', 'L', 'R'}


# In[4]:


def random_action(a, eps=0.1):
    '''We will use epsilon soft to ensure that all the states are visited. If set eps=0 some states may never be visited. '''
    p = np.random.random()
    if p < (1 - eps):
        return a
    else:
        return np.random.choice(list(ALL_POSSIBLE_ACTION))


# In[5]:


def play_game(grid, policy):
    '''Returns a List of states and corresponding returns
    Reset the Game to start at a random position, we need to do this beacuse given our current determinisitc policy, 
    we would never end up at certain states, but we still want to measure it.(exploring starts method.)'''
    s = (2, 0)
    grid.set_state(s)    
    states_and_rewards = [(s, 0)] #for a starting state we will give a reward of 0.
    while not grid.game_over():
        a = policy.get(s)        
        a = random_action(a)
        r = grid.move(a)
        s = grid.current_state()
        states_and_rewards.append((s, r))
    return states_and_rewards


# In[6]:


def main():
    #using the standard grid again (0 for every step) so that we can compare to iterative policy evaluation.
    grid = standard_grid()
    
    #print rewards
    print_values(grid.rewards, grid)
    
    #Policy - For a given state what is the action we would take.
    #set the initial policy to random actions
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
    
    #Initialize V(s)
    V = {}
    states = grid.all_states()
    for s in states:
        V[s] = 0
    #Repeat untill Covergence 
    for it in range(1000):
        states_and_rewards = play_game(grid, policy)
        #The first (s, r) tuple is the state we start in and 0(since we don't get a reward for simply starting the game)
        #the last (s, r) tuple is the terminal state and the final reward. the value of a terminal state by definition is 0, 
        #so we don't care about updating it.
        
        for t in range(len(states_and_rewards) - 1):
            s, _ = states_and_rewards[t]
            s2, r = states_and_rewards[t + 1]
            
            #We will update V[s] as we will experience the episode.
            V[s] = V[s] + ALPHA * (r + (GAMMA * V[s2]) - V[s])
        
        
    print("values: ")
    print_values(V, grid)
    
    print("policy: ")
    print_policy(policy, grid)
    


# In[7]:


if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




