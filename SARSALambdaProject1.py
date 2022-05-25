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
wall = [31, 32, 33, 34]

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

# this below code takes cells that have wall in it
# and places a value of 13 in the cell.  

for s in range(nOS):
    if s in wall:
        best_value[s] = 13
            
print("Time Taken till Best Policy After Value Iteration")
print(time_taken)        

print("Best Policy After Value Iteration")        
print(show_best_policy.reshape(6,6))

print("Best Policy Values After Value Iteration")
print (best_value.reshape(6,6))



reshaped_best_value = best_value.reshape(6,6)


# below function is being used to assign shi-greedy action
# to Q learning algorithm.  Note that because this is first
# action dictionary above addresses the challenge of not
# allowing exits from cells on the outside faces


def greedy(epsilon, Q_s):
    num1 = random.randrange(0,10)
    if num1 <10*epsilon:
        return random.randrange(nOA)
    else:
         
        return np.argmax(Q_s)


# below function is being used to assign shi-greedy for the second action
# in SARSA.  Note that this code addresses the problem of not allowing actions
# that allow agent to exit the grid from the outside faces.
# it also allows for allowing the agent to perform exploration-exploitation
# until 20,000 episodes and after that the agent simply explits as epsilon
# drops to a very small amount
# not that the code does index slicing and renumbering to not allow 
# unpermitted actions. ignoring transitioning out of shi/k decay
    
    
def greedy1(epsilon, Q_s, episode, s):
    num1 = random.randrange(0,10)
    if episode >20000:
        if s <6:
            j= np.argmax(Q_s[1:])+1
        
        if s ==5:
           c = Q_s[2:]
           
           j = np.argmax(c)+2 
        
        if (s+1)%6 == 0 and s != 5:
            c = (Q_s[:1])
            cc = np.zeros((1))
            cc[0]=-999999
            d = (Q_s[2:])
            e = np.concatenate((c,cc, d))
            j = np.argmax(e)
        if (36-s)<=5:
            c = (Q_s[:2])
            cc = np.zeros((1))
            cc[0]=-999999
            
            d = (Q_s[3:])
            e = np.concatenate((c, cc, d))
            j = np.argmax(e)
        
            
        if s == 30:
           c = Q_s[:2]
           j = np.argmax(c)
            
        if (s)%6 ==0 and s != 30:
            j = np.argmax(Q_s[:3])
        if s >5:
           if s <30:
               if (s)%6 != 0:
                   if (s+1)%6 !=0: 
                       j = np.argmax(Q_s)
    else:
    
        if num1 <10*epsilon:
            if s <6:
                j= random.randrange(1,4)
            if (s+1)%6 == 0:
                
                j = (choice([i for i in range(0,4) if i not in [1]]))
            if (36-s)<=6:
                
                j = (choice([i for i in range(0,4) if i not in [2]]))
            if (s)%6 ==0:
                j = (choice([i for i in range(0,4) if i not in [3]]))
            else:
                j = random.randrange(nOA)
        else:
            if s <6:
                j= np.argmax(Q_s[1:])+1
            
            if s ==5:
               c = Q_s[2:]
               
               j = np.argmax(c)+2 
            
            if (s+1)%6 == 0 and s != 5:
                c = (Q_s[:1])
                cc = np.zeros((1))
                cc[0]=-999999
                d = (Q_s[2:])
                e = np.concatenate((c,cc, d))
                j = np.argmax(e)
            if (36-s)<=5:
                c = (Q_s[:2])
                cc = np.zeros((1))
                cc[0]=-999999
                
                d = (Q_s[3:])
                e = np.concatenate((c, cc, d))
                j = np.argmax(e)
            
                
            if s == 30:
               c = Q_s[:2]
               j = np.argmax(c)
                
            if (s)%6 ==0 and s != 30:
                j = np.argmax(Q_s[:3])
            if s >5:
               if s <30:
                   if (s)%6 != 0:
                       if (s+1)%6 !=0: 
                           j = np.argmax(Q_s)
    return j 
     




# SARSA lambda algorithm


# this below code allows implementation of SARSA lambda
# one starting state is randomly picked



def SARSAlambda(no_episodes,lamb, epsilon, no_steps, learn_rate, discount):

    Q = np.zeros((nOS, nOA))
    RSARSA_List = []
    fails=0
    no_states = []
    
    
         
    
    
    State = random.randrange(nOS)
    
    for episode in range(no_episodes):
        
        eligib_traces = np.zeros((nOS, nOA))    
        s= State
        
 #Note that this code addresses the problem of not allowing actions
 # that allow agent to exit the grid from the outside faces.
 # it also allows for allowing the agent to perform exploration-exploitation
# note that the code does index slicing and renumbering to not allow 
 # unpermitted actions        
        
        
        if s <6:
            a= random.randrange(1,4)
        if (s+1)%6 == 0:
            
            a = (choice([i for i in range(0,4) if i not in [1]]))
        if (36-s)<=6:
            
            a = (choice([i for i in range(0,4) if i not in [2]]))
        if (s)%6 ==0:
            a = (choice([i for i in range(0,4) if i not in [3]]))
        else:
            a = random.randrange(nOA)
       
        step = 0
       
        R =0
            
        while step <no_steps and terminal_state(s) != True:
            if episode ==0:
                epsilon1 = epsilon
            else:
                epsilon1 = epsilon/(episode)
                    
            A = a
            
            next_state, probability, reward = P[s][A]
             
            
            #print(epsilon1)
            A_ = greedy1(epsilon1, Q[next_state], episode, s)
            #A_ = np.argmax(Q[next_state])
            
            delta = reward + discount*Q[next_state][A_] - Q[s][A]
            eligib_traces[s][A] = eligib_traces[s][A]+1
            
            for stat in range(nOS):
                for Act in range (nOA):
                    if terminal_state(stat) != True :
                        Q[stat][Act] = Q[stat][Act] + learn_rate * delta * eligib_traces[stat][Act]
                        eligib_traces[stat][Act] = discount*lamb*eligib_traces[next_state][A_]
            
            s = next_state
            a = A_ 
           
            step += 1
            R = R + reward
    
        RSARSA_List.append(R)
        no_states.append(step)
                
        if terminal_state(s) == False:
            fails = fails +1
          
    
    print(fails)
        
    return Q, RSARSA_List, no_states

resultsarsaQ, RSARSA_List, no_states = SARSAlambda(5000, 0.4, .1, 200, 0.2, 0.9)



print(resultsarsaQ)

bestQSARSA = np.zeros((nOS))

best_SARSApolicy = np.zeros((nOS))

# the below code is used to generate optimum policy from calculated Q(s,a) values


for s in range(nOS):
    bestQSARSA[s] = max(resultsarsaQ[s])
    
    if terminal_state(s):
        best_SARSApolicy[s] = 4
    
    elif s in wall:
        best_SARSApolicy[s] = 5
        
    else:
        if s <5:
            best_SARSApolicy[s]= np.argmax(resultsarsaQ[s][1:])+1
        if s ==5:
            c = resultsarsaQ[s][2:]
            best_SARSApolicy[s] =  np.argmax(c)+2
        if s!= 5:
            if (s+1)%6 ==0:
                c = (resultsarsaQ[s][:1])
                cc = np.zeros((1))
                cc[0]=-999999
                
                d = (resultsarsaQ[s][2:])
                e = np.concatenate((c,cc,d))
                best_SARSApolicy[s] = np.argmax(e)
        if (36-s)<=5:
            c = (resultsarsaQ[s][:2])
            cc = np.zeros((1))
            cc[0]=-999999
            
            d = (resultsarsaQ[s][3:])
            e = np.concatenate((c,cc,d))
            
            best_SARSApolicy[s] = np.argmax(e)
        if s==30:
            c = resultsarsaQ[s][:2]
            best_SARSApolicy[s] = np.argmax(c)
        if s!= 30:
            if (s)%6 == 0:
                best_SARSApolicy[s] = np.argmax(resultsarsaQ[s][:3])
        if s >5:
            if s <30:
                if (s)%6 != 0:
                    if (s+1)%6 !=0: 
                        #print(s)
                        best_policy[s] = np.argmax(resultsarsaQ[s])


for s in range(nOS):
    if s in wall:
        bestQSARSA[s] = 13
        
final_result = bestQSARSA.reshape(6,6)
print(final_result)


best_SARSA_policy = best_SARSApolicy.reshape(6,6)
print(best_SARSA_policy)


# the below code is used to generate RMSE between SARSA lambda value computation
# and optimum values estimated from value iteration

rmse =0
n=0
for i in range(6):
    for j in range(6):
        
        n = n+1
        
        rmse +=(final_result[i][j]- reshaped_best_value[i][j])**2
rmse = (rmse/36)**0.5

print(rmse)
        