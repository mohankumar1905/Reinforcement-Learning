#!/usr/bin/env python
# coding: utf-8

# In[21]:


import numpy as np
from matplotlib import pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy


# In[22]:


SMALL_ENOUGH = 10e-4
GAMMA = .9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


# This is deterministic all p(s', r|s, a) = 1 or 0

# In[38]:


def main(grid_name="standard"):
    '''This grid gives you a neagative reward of -0.1 for every non terminal state. We want to see, whether it will envourage an
    agent to find shorter path to the goal'''
    if grid_name == "negative":
        grid = negative_grid()

        #print rewards
        print("rewards: ")
        print_values(grid.rewards, grid)
    else:
        grid = standard_grid()

    #Creating a Deterministic random policy which maps each state to a random action.
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)


    print("Initial Policy: ")
    print_policy(policy, grid)

    #initialize V(s)
    states = grid.all_states()
    V = {}
    for s in states:
        if s in grid.actions:
            V[s] = np.random.random() #generates a random number between zero to one.
        else:
            V[s] = 0

    #Repeating untill convergence - Breaks out when policy does not change.
    while True:

        #policy evaluation step - we already know how to do this.
        while True:
            biggest_change = 0
            for s in states:
                old_v = V.get(s)

                #V(s) has a value only for a non terminal state
                if s in policy:
                    grid.set_state(s)
                    a = policy.get(s)
                    r = grid.move(a)
                    V[s] = r + (GAMMA * V.get(grid.current_state()))
                    biggest_change = max(biggest_change, np.abs(old_v - V.get(s)))

            if biggest_change < SMALL_ENOUGH:
                break

        #Policy Improvement Step
        is_policy_converged = True
        for s in states:
            if s in policy:
                old_a = policy.get(s)
                new_a = None
                best_value = float('-inf') #assiging it to lowest possible value -infinity
                #loop through all possible action to find best possible action
                for a in ALL_POSSIBLE_ACTIONS:
                    grid.set_state(s)
                    r = grid.move(a)
                    v = r + GAMMA * (V.get(grid.current_state()))
                    if v > best_value:
                        best_value = v
                        new_a = a
                policy[s] = new_a
                if new_a != old_a:
                    is_policy_converged = False

        if is_policy_converged:
            break

    print("Values: ")
    print_values(V, grid)

    print("Policy: ")
    print_policy(policy, grid)


# In[41]:


if __name__ == '__main__':
    main("negative")
    print("-" * 100)
    main()


# In[ ]:




