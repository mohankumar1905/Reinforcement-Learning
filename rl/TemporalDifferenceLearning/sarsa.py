#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("C:\\Users\\mohan\\Documents\\GitHub\\Reinforcement-Learning\\rl\\DynamicProgramming")


# In[2]:


sys.path.append("C:\\Users\\mohan\\Documents\\GitHub\\Reinforcement-Learning\\rl\\MonteCarlo")


# In[3]:


import numpy as np
from matplotlib import pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy
from monte_carlo_exploring_starts import max_dict
from td0_prediction import random_action


# In[4]:


ALPHA = 0.1
GAMMA = 0.9
ALL_POSSIBLE_ACTION = {'U', 'D', 'L', 'R'}


# In[5]:


def main():
    #using the standard grid again (0 for every step) so that we can compare to iterative policy evaluation.
    grid = negative_grid()
    
    #print rewards
    print_values(grid.rewards, grid)
    
    #Initialize V(s)
    Q = {}
    states = grid.all_states()
    for s in states:
        Q[s] = {}
        for a in ALL_POSSIBLE_ACTION:
            Q[s][a] = 0
            
            
    #lets keep track of how many times Q[s] has been updated.
    update_counts = {} #what portion of time we spend in each state. - for debugging
    update_counts_sa = {} #this is for adaptive learning rate
    for s in states:
        update_counts_sa[s] ={}
        for a in ALL_POSSIBLE_ACTION:
            update_counts_sa[s][a] = 1.0
            
    t = 1.0
    deltas = []
    for it in range(10000):
        if it%100 == 0:
            t += 10e-3
        if it%2000 == 0:
            print(it)
        #instead of generating the episode we will play an episode within the loop.
        s = (2, 0)
        grid.set_state(s)

        #The first (s, r) tuple is the state we start in and 0(since we don't get a reward for simply starting the game)
        #the last (s, r) tuple is the terminal state and the final reward. the value of a terminal state by definition is 0, 
        #so we don't care about updating it.
        a = max_dict(Q[s])[0]
        a = random_action(a, eps=0.5/t)
        biggest_change = 0
        
        while not grid.game_over():
            r = grid.move(a)
            s2 = grid.current_state()
            #we need next action as well since Q(a, s) depends on Q(s', a'), If s2 not in policy then it is a terminal state
            a2 = max_dict(Q[s2])[0]
            a2 = random_action(a2, eps=0.5/t)
            
            #we will update Q(s, a) as we experience the episode.
            alpha = ALPHA/ update_counts_sa[s][a]
            update_counts_sa[s][a] += 0.005 #updating the count only by a small amount
            old_qsa = Q[s][a]
            Q[s][a] = Q[s][a] + alpha * (r + (GAMMA * Q[s2][a2]) - Q[s][a])
            biggest_change = max(biggest_change, np.abs(old_qsa - Q[s][a]))
            
            #we would like to know how often Q[s] has been updated
            update_counts[s] = update_counts.get(s, 0) + 1
            
            s = s2
            a = a2
        deltas.append(biggest_change)
    plt.plot(deltas)
    plt.show()
    
    #Determine the policy Q* from V*
    policy = {}
    V = {}
    for s in grid.actions.keys():
        a, max_q = max_dict(Q[s])
        policy[s] = a
        V[s] = max_q

        
    #what portion of time we spend updating each part of q.
    print ("update counts:")
    total = np.sum(list(update_counts.values()))
    for k, v in update_counts.items():
        update_counts[k] = v/total
    print_values(update_counts, grid)
    
    
    print("values: ")
    print_values(V, grid)
    
    print("policy: ")
    print_policy(policy, grid)
    


# In[6]:


if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




