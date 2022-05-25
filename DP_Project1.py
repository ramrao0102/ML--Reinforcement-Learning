# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 10:18:30 2022
@author: ramra
"""

# Ram Rao Reinforcement Learning Project

# The problem to be solved is a gridworld 6x6 maze
# The number of states in the grid are 36, from 0 to 35

# No of actions in each state are 4
# Index of Actions
#0: up
#1: right
#2: down
#3: left
# note that policy actions for wall and terminal state are also provided below
#4: terminal state
#5: wall

import numpy as np
import pandas as pd
import time
import math
import random
import matplotlib.pyplot as plt
from random import choice


# number of states
nOS = 6*6

# number of states are 36 and go from 0 to 5 on top row
#6 to 11 on row below it and so on

np.set_printoptions(precision = 2)
S = range(nOS)

# number of actions
nOA = 4

# defining action values
up = 0
right = 1
down = 2
left = 3

reward = -1
reward_terminal = 0
reward_wall = -1000

# creating a list for states that have a wall
wall = []

# function for checking if state is terminal state or not 

def terminal_state(s):
    if s ==0 or s == nOS -1:
        return True
    else:
        return False


#creating a dictionary of next_state, transition_probability, and reward for taking action
# each value will be a tuple of above values 

P = dict()

for s in range(nOS):
    P[s] = dict()
    if (terminal_state(s)):
        P[s][up] = (s, 1.0,reward_terminal)
        P[s][right] = (s,1.0, reward_terminal)
        P[s][down] = (s,1.0, reward_terminal)
        P[s][left] = (s, 1.0, reward_terminal)
        
    else:
        if s <6:
            next_state = s
        else:
            next_state = s-6
            
        if next_state in wall:
            P[s][up] = (next_state, 1.0, reward_wall)
        else:
            P[s][up] = (next_state, 1.0, reward)
        
        if (s+1)%6 == 0:
            next_state = s
        else:
            next_state = s+1
        
        if next_state in wall:
            P[s][right] = (next_state, 1.0, reward_wall)
        else:
            P[s][right] = (next_state, 1.0, reward)
            
        if (36-s)<=6:
            next_state = s
        else:
            next_state = s+6
        
        if next_state in wall:
            P[s][down] = (next_state, 1.0, reward_wall)
        else:
            P[s][down] = (next_state, 1.0, reward)
            
        if (s)%6 ==0:
            next_state=s
        else:
            next_state = s-1
            
        if next_state in wall:
            P[s][left] = (next_state, 1.0, reward_wall)
        else:
            P[s][left] = (next_state, 1.0, reward)


# Function for Policy Evaluation

def Policy_Evaluation(Policy, threshold, discount):
    
    # initalizing the value initially to zero
    value = np.zeros(nOS, )
    
    while True:
        delta = 0
        
        for s in range(nOS):
            v= value[s]
            temp = 0
            for a, action_policy_probability in enumerate(Policy[s]):
                next_state, probability, reward = P[s][a]
               
                temp += action_policy_probability*probability*(reward + discount*value[next_state]) 
                
            value[s] = temp
           
            delta=max(delta,np.abs(v-value[s]))  
                
        if delta <threshold:
            break
            
    return value
            
# testing policy evaluation with equi probable policy

random_policy = np.ones([36,4])/4
threshold  = 0.0001
discount = 1.0

random_policy_value =   Policy_Evaluation(random_policy, threshold, discount)

print("Equi-Probable Policy Values")
print(random_policy_value.reshape(6,6))

# Function for Policy Iteration

def Policy_Iteration(discount, threshold):
    value = np.zeros(nOS,)
    policy = np.ones([36,4])/4
    to = time.perf_counter()
    
    while True:
        value =Policy_Evaluation(policy, threshold, discount)
        new_policy = np.zeros([nOS, nOA])
        policy_stable = True
        
        for s in range(nOS):
            if (terminal_state(s) != True):
                previous_policy = policy[s]
                state_action_values = np.zeros(nOA)
                
                for a in range(nOA):
                    next_state, probability, reward = P[s][a]
                    state_action_values[a] += probability*(reward+discount*value[next_state])
                    
                    
                max_total = np.amax(state_action_values)
                best_a = np.argmax(state_action_values)
                
                new_policy[s][best_a]=1
                value[s] = max_total
                
                if (np.array_equal(previous_policy, new_policy[s]) != True):
                    policy_stable = False
                    
        if policy_stable:
            t1 = time.perf_counter() - to
            return new_policy, value, t1
        
        else:
            policy= new_policy
            
    
            
                
# Print out Policy and Value After Policy Iteration

stable_policy, stable_value, time_taken =  Policy_Iteration(discount, threshold)

show_stable_policy =   np.zeros(nOS,)

for s, prob in enumerate(stable_policy):
    
    if terminal_state(s):
        show_stable_policy[s] = 4
    
    elif s in wall:
        show_stable_policy[s] = 5
        
    else:
        show_stable_policy[s] = np.argmax(prob)

print("Time Taken till Stable Policy After Policy Iteration")
print(time_taken)

print("Stable Policy After Policy Iteration")        
print(show_stable_policy.reshape(6,6))

print("Stable Policy Values After Policy Iteration")
print (stable_value.reshape(6,6))

# Function for Value Iteration

def Value_Iteration(discount, threshold):
    value = np.zeros(nOS,)
    to = time.perf_counter()
    while True:
        delta = 0
        
        new_policy = np.zeros([nOS,4])
        
        for s in range(nOS):
        
            if (terminal_state(s) != True):
                v = value[s]
                state_action_values = np.zeros(nOA)
            
                for a in range(nOA):
                    next_state, probability, reward = P[s][a]
                    state_action_values[a] += probability*(reward+discount*value[next_state])
                                    
                max_total = np.amax(state_action_values)
                best_a = np.argmax(state_action_values)
             
                value[s] = max_total
                                
                new_policy[s][best_a]=1
                delta=max(delta,np.abs(v-value[s]))
                
                
        if delta < threshold:
            break
    
    t1 = time.perf_counter() - to    
    return new_policy, value, t1


# Print out Policy and Value After Value Iteration

best_policy, best_value, time_taken =  Value_Iteration(discount, threshold)
   
show_best_policy =   np.zeros(nOS,)

for s, prob in enumerate(best_policy):
    
    if terminal_state(s):
        show_best_policy[s] = 4
    
    elif s in wall:
        show_best_policy[s] = 5
        
    else:
        show_best_policy[s] = np.argmax(prob)
        
print("Time Taken till Best Policy After Value Iteration")
print(time_taken)        

print("Best Policy After Value Iteration")        
print(show_best_policy.reshape(6,6))

print("Best Policy Values After Value Iteration")
print (best_value.reshape(6,6))



        