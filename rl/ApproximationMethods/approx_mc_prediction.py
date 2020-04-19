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
from monte_carlo_random import random_action, play_game, SMALL_ENOUGH, GAMMA


# In[7]:


LEARNING_RATE = 0.001


# In[28]:


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
        (1, 2): 'U',
        (2, 1): 'L',
        (2, 2): 'U',
        (2, 3): 'L'
    }
    
    #Initialize theta - our model is V_hat = theta.dot(x)
    theta = np.random.randn(4)/2
    
    #where x is [row, column, row*column, 1] - for bias term
    def state_to_feature(s):
        return np.array([s[0] -1, s[1] - 1.5, s[0] * s[1] - 3, 1])
    
    deltas = []
    t = 1.0
    for it in range(20000):
        if it % 100 == 0:
            t += 0.01
        alpha = LEARNING_RATE/t
        #generate an episode using given policy
        biggest_change = 0
        states_and_returns = play_game(grid, policy)
        seen_states = set()
        for s, G in states_and_returns:
            if s not in seen_states:
                old_theta = theta.copy()
                x = state_to_feature(s)
                V_hat = theta.dot(x)
                #grad V_hat with respect to theta = x
                theta += alpha * (G - V_hat) * x
                biggest_change = max(biggest_change, np.abs(theta - old_theta).sum())
                seen_states.add(s)
        deltas.append(biggest_change)
    plt.plot(deltas)
    plt.show()

    #obtain predicted values
    V = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            V[s] = theta.dot(state_to_feature(s))
        else:
            V[s] = 0
    print("Values: ")
    print_values(V, grid)

    print("Policy: ")
    print_policy(policy, grid)


# In[29]:


if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




