#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
from grid_world import standard_grid


# In[2]:


SMALL_ENOUGH = 10e-4


# In[25]:


def print_values(V, g):
    '''It takes Values dictionary and Grid g(object)'''
    for i in range(g.width):
        print("-"*20)
        for j in range(g.height):
            v = V.get((i, j), 0)
            if v>= 0:
                print(f"  {round(v, 2)}", end="")
            else:
                print(f" {round(v, 2)}", end="") #-ve sign takes up the extra space
        print("")       


# In[20]:


def print_policy(P, g):
    '''it takes policy P as an input. It will only work for deterministic policies'''
    for i in range(g.width):
        print("-"*20)
        for j in range(g.height):
            a = P.get((i, j), ' ')
            print(f" {a} |", end="")
        print("")       


# In[26]:


def main():
    '''Iterative policy evaluation: Given a policy, let's find its value function.
    We will do this for both a uniform random policy and fixed policy.
    Note - there are two sources of randomness.
    p(a/s) -> deciding what action to take given the state.
    p(s', r/ s, a) -> the next state and reward given your state-action pair
    we are only modeling p(a/s) = uniform  '''

    grid = standard_grid()

    #states will be positions(i, j), simpler than tic-tac-toe, because we only have one game piece.
    #so there can be only one position at a time

    states = grid.all_states()

    #uniformly random actions
    #initialize V(s) = 0
    V = {}
    for s in states:
        V[s] = 0
    gamma = 1 #growth factor

    while True:
        biggest_change = 0 #Delta
        for s in states:
            old_v = V.get(s)

            #V(s) has value only if it  is not in a terminal state
            if s in grid.actions:
                new_v = 0 #For accumulating v over all possible actions
                p_a = 1.0/len(grid.actions.get(s))
                for a in grid.actions.get(s):
                    grid.set_state(s)
                    r = grid.move(a)
                    new_v += p_a * (r + gamma* V.get(grid.current_state()))
                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(new_v - old_v))
        if biggest_change < SMALL_ENOUGH:
            break
    print("Values of each state for a uniform random actions: ")
    print_values(V, grid)
    print()
    print()
    
    ###fixed policy###
    policy = {
        (2, 0): 'U',
        (1, 0): 'U',
        (0, 0): 'R',
        (0, 1): 'R',
        (0, 2): 'R',
        (2, 1): 'R',
        (2, 2): 'R',
        (2, 3): 'U',
        (1, 2): 'R'
    }
    print_policy(policy, grid)
    V = {}
    for s in states:
        V[s] = 0
    gamma = 0.9 #growth factor

    while True:
        biggest_change = 0 #Delta
        for s in states:
            old_v = V.get(s)

            #V(s) has value only if it  is not in a terminal state
            if s in policy:
                new_v = 0 #For accumulating v over all possible actions
                p_a = 1.0
                a = policy.get(s)
                grid.set_state(s)
                r = grid.move(a)
                new_v += p_a * (r + gamma* V.get(grid.current_state()))
                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(new_v - old_v))
        if biggest_change < SMALL_ENOUGH:
            break
    print("Values of each state for a fixed policy ")
    print_values(V, grid)
    
if __name__ == '__main__':
    main()


# In[ ]:




