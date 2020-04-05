#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt


# In[9]:


class Grid:
    '''Environment class'''
    def __init__(self, width, height, start):
        '''current location will be stored in the instance variable i and j will be at start'''
        self.width = width
        self.height = height
        self.i = start[0]
        self.j = start[1]
        
    def set(self, rewards, actions):
        '''Rewards should be a dict of (i,j) : reward value. Actions should be a dict of (i, j): action list possible'''
        self.rewards = rewards
        self.actions = actions
        
    def set_state(self, s):
        '''Initialize the state given i and j'''
        self.i = s[0]
        self.j = s[1]
        
    def current_state(self):
        '''Returns the current state i, j position of an agent'''
        return (self.i, self.j)
    
    def is_terminal(self, s):
        '''given a state if there are no possible actions available in that state, then it is a terminal state'''
        return s not in self.actions
    
    def move(self, action):
        '''it moves an agent to another grid'''
        if action in self.actions[(self.i, self.j)]:
            if action == 'U':
                self.i -= 1
            elif action == 'D':
                self.i += 1
            elif action == 'R':
                self.j += 1
            elif action == 'L':
                self.j -= 1
        return self.rewards.get((self.i, self.j), 0) #if there is no reward to find for this position the default is zero.
    
    def undo_move(self, action):
        '''If you pass the action you just did, the environment will undo it'''
        if action == 'U':
            self.i += 1
        elif action == 'D':
            self.i -= 1
        elif action == 'R':
            self.j -= 1
        elif action == 'L':
            self.j += 1
        #raise an exception if we arrive somewhere we shouldn't be. It should never happen. 
        assert(self.current_state() in self.all_states())
        
    def game_over(self):
        '''returns true if a game is over, else false
        True - if we are in a state where no actions are possible'''
        return (self.i, self.j) not in self.actions
    
    def all_states(self):
        '''Possibly buggy but a simple way to get all states, either a position has a next possible action or a position that
        yeilds reward'''
        return set(list(self.actions.keys()) + list(self.rewards.keys()))


# In[11]:


def standard_grid():
        '''Define a grid that describes the reward for arriving at each state and possible action at each state.
        The grid look like below . x means you can't go there, s means start, number is the reward you get in that state 
        .  .  . -1
        .  x  .  1
        s  .  .  .
        '''
        g = Grid(3, 4, (2, 0))
        rewards = {(0, 3): 1, (1, 3): -1}
        actions = {
            (0, 0): ('D', 'R'),
            (0, 1): ('L', 'R'),
            (0, 2): ('L', 'R', 'D'),
            (1, 0): ('U', 'D'),
            (1, 2): ('U', 'D', 'R'),
            (2, 0): ('U', 'R'),
            (2, 1): ('L', 'R'),
            (2, 2): ('L', 'R', 'U'),
            (2, 3): ('L', 'U')
        }
        g.set(rewards, actions)
        return g


# In[12]:


def negative_grid(step_cost = -.1):
    g = standard_grid()
    g.rewards.update({
        (0, 0): step_cost,
        (0, 1): step_cost,
        (0, 2): step_cost,
        (1, 0): step_cost,
        (1, 2): step_cost,
        (2, 0): step_cost,
        (2, 1): step_cost,
        (2, 2): step_cost,
        (2, 3): step_cost,
    })
    return g



