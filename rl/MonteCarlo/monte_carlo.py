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


SMALL_ENOUGH  = 10e-4
GAMMA = 0.9
ALL_POSSIBLE_ACTION = {'U', 'D', 'L', 'R'}


# In[31]:


def play_game(grid, policy):
    '''Returns a List of states and corresponding returns
    Reset the Game to start at a random position, we need to do this beacuse given our current determinisitc policy, 
    we would never end up at certain states, but we still want to measure it.'''
    start_states = grid.actions.keys() #start states can be anything except terminal state where we cannot take action.
    start_idx = np.random.choice(len(start_states))
    grid.set_state(list(start_states)[start_idx])
    
    s = grid.current_state()
    states_and_rewards = [(s, 0)] #for a starting state we will give a reward of 0.
    while not grid.game_over():
        a = policy.get(s)
        r = grid.move(a)
        s = grid.current_state()
        states_and_rewards.append((s, r))

    #Calculate the returns by working backwards from the terminal state.
    G = 0 #the value of a terminal state is zero by definition
    states_and_returns = []
    first = True #True because we are calculating first visit monte carlo
    
    for s, r in reversed(states_and_rewards):
        if first:
            first = False
        else:
            states_and_returns.append((s, G))
        G = r + (GAMMA * G)
    states_and_returns.reverse()
    return states_and_returns


# In[34]:


def main():
    #using the standard grid again (0 for every step) so that we can compare to iterative policy evaluation.
    grid = standard_grid()
    
    #print rewards
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
    
    #Initialize V(s) and returns 
    V = {}
    returns = {} #dictionary of a state -> list of returns we have recieved.
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            returns[s] = []
        else:
            V[s] = 0
    
    #Repeat 
    for t in range(100):
        #generate an episode using pi
        states_and_returns = play_game(grid, policy)
        seen_states = set()
        for s, G in states_and_returns:
            #We are calculating using first visit monte carlo method. - so we check whether we have seen the state already 
            #monte-carlo method.
            if s not in seen_states:
                returns[s].append(G)
                V[s] = np.mean(returns[s])
                seen_states.add(s)
                
    print("values: ")
    print_values(V, grid)
    print("policy: ")
    print_policy(policy, grid)
        


# In[35]:


if __name__ == "__main__":
    main()


# In[ ]:




