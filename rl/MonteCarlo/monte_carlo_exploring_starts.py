#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''Included Control problem- This scripts implement Monte carlo exploring starts method for finding optimal policy.'''


# In[2]:


import sys
sys.path.append("C:\\Users\\mohan\\Documents\\GitHub\\Reinforcement-Learning\\rl\\DynamicProgramming")


# In[3]:


import numpy as np
from matplotlib import pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy


# In[4]:


GAMMA = 0.9
ALL_POSSIBLE_ACTION = {'U', 'D', 'L', 'R'}


# In[5]:


ALL_POSSIBLE_ACTION


# In[6]:


def random_action(a):
    '''Choose given "a" with probablity 0.5 and some other a'!=a with probablity 0.5/3 '''
    p = np.random.random()
    if p < 0.5:
        return a
    else:
        temp = ALL_POSSIBLE_ACTION.copy()
        temp.remove(a)
        return np.random.choice(list(temp))


# In[7]:


def play_game(grid, policy):
    '''Returns a List of states and corresponding returns
    Reset the Game to start at a random position, we need to do this beacuse given our current determinisitc policy, 
    we would never end up at certain states, but we still want to measure it.(exploring starts method.)'''
    start_states = grid.actions.keys() #start states can be anything except terminal state where we cannot take action.
    start_idx = np.random.choice(len(start_states))
    grid.set_state(list(start_states)[start_idx])
    
    s = grid.current_state()
    a = np.random.choice(list(ALL_POSSIBLE_ACTION))
    
    states_action_rewards = [(s, a, 0)] #for a starting state we will give a reward of 0.
    stop_itertion = False
    seen_states = set()
    seen_states.add(grid.current_state())
    num_steps = 0
    while not stop_itertion:
        num_steps+=1
        r = grid.move(a)
        s = grid.current_state()
        if s in seen_states:
            rewards= -2/num_steps
            states_action_rewards.append((s, None, -100))
            stop_itertion = True
        elif grid.game_over():
            states_action_rewards.append((s, None, r))
            stop_itertion = True
        else:
            a = policy.get(s)        
            a = random_action(a)
            states_action_rewards.append((s, a, r))
        seen_states.add(s)

    #Calculate the returns by working backwards from the terminal state.
    G = 0 #the value of a terminal state is zero by definition
    states_actions_returns = []
    first = True #True because we are calculating first visit monte carlo

    for s, a, r in reversed(states_action_rewards):
        if first:
            first = False
        else:
            states_actions_returns.append((s, a, G))
        G = r + (GAMMA * G)
    states_actions_returns.reverse()
    return states_actions_returns


# In[8]:


def max_dict(d):
    '''Returns the argmax(key) and max(value) from a dictionary'''
    max_key = None
    max_val = float('-inf')
    for k, v in d.items():
        if v > max_val:
            max_val = v
            max_key = k
    return max_key, max_val


# In[9]:


def main():
    #using the standard grid again (0 for every step) so that we can compare to iterative policy evaluation.
    grid = negative_grid()
    
    #print rewards
    print_values(grid.rewards, grid)
    
    #Policy - For a given state what is the action we would take.
    #set the initial policy to random actions
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(list(ALL_POSSIBLE_ACTION))

    #Initialize V(s) and returns 
    Q = {}
    returns = {} #dictionary of a state -> list of returns we have recieved.
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            Q[s] = {}
            for a in ALL_POSSIBLE_ACTION:
                Q[s][a] = 0
                returns[(s, a)] = []
        else:
            pass #terminal state or state we can't otherwise get to.

    deltas = []
    for t in range(2000):
        if t % 100 == 0:
            print(t)
        
        biggest_change = 0
        states_actions_returns = play_game(grid, policy)
        seen_state_action_pairs = set()
        for s, a, G in states_actions_returns:
            #We are calculating using first visit monte carlo method. - so we check whether we have seen the state action already 
            sa = (s, a)
            if sa not in seen_state_action_pairs:
                old_q = Q[s][a]
                returns[sa].append(G)
                Q[s][a] = np.mean(returns[sa])
                biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))
                seen_state_action_pairs.add(sa)
        deltas.append(biggest_change)
    
        #update policy
        for s in policy.keys():
            policy[s] = max_dict(Q[s])[0]
        
    plt.plot(deltas)
    plt.show()
    
    print("final policy: ")
    print_policy(policy, grid)
    
    #find V  
    V = {}
    for s, Qs in Q.items():
        V[s] = max_dict(Q[s])[1]
        
    print("values: ")
    print_values(V, grid)


# In[10]:


if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




