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

for s in range(nOS):
    if s in wall:
        stable_value[s] =13

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

# this below code takes cells that have wall in it
# and places a value of 13 in the cell.  

for s in range(nOS):
    if s in wall:
        best_value[s] =13

print("Best Policy Values After Value Iteration")
print (best_value.reshape(6,6))

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
# unpermitted actions
    
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
     

# Q LEARNING ALGORITHM

def Qlearning(no_episodes,no_steps, epsilon, learn_rate, discount):
    Q = np.zeros((nOS, nOA))
    RQ_List = []
    fails =0
    no_statesQ = []

# this below code allows implementation of Q learning
# a starting state is creating from 1 to 33, and if not in wall
# process below is performed and we track the negative reward and # of states 
# visited for each episode/state

    
    state=1
    while state <33 and state not in wall:
       
        for episode in range(no_episodes):
            
            s= state
        
            step = 0
            R =0
            while step <no_steps and terminal_state(s) != True:
                 
                    
                A = greedy(epsilon, Q[s])
             
                next_state, probability, reward = P[s][A]
            
#Note that this code addresses the problem of not allowing actions
# that allow agent to exit the grid from the outside faces.
# it also allows for allowing the agent to perform exploration-exploitation
# not ethat the code does index slicing and renumbering to not allow 
# unpermitted actions
            
            
                if  next_state <5:
                    A_= np.argmax(Q[next_state][1:])+1
                
                if  next_state ==5:
                   c = Q[next_state][2:]
                   
                   A_ = np.argmax(c)+2 
                
                if ( next_state+1)%6 == 0 and  next_state != 5:
                    c = (Q[next_state][:1])
                    cc = np.zeros((1))
                    cc[0]=-999999
                    
                    d = (Q[next_state][2:])
                    e = np.concatenate((c,cc,d))
                    A_ = np.argmax(e)
                if (36- next_state)<=5:
                    c = (Q[next_state][:2])
                    cc = np.zeros((1))
                    cc[0]=-999999
                    
                    d = (Q[next_state][3:])
                    e = np.concatenate((c,cc,d))
                    A_ = np.argmax(e)
                
                if  next_state == 30:
                   c = Q[next_state][:2]
                   A_ = np.argmax(c)
                
                if ( next_state)%6 == 0 and s!= 30:
                    A_ = np.argmax(Q[next_state][:3])
                if  next_state >5:
                    if  next_state <30:
                        if ( next_state)%6 != 0:
                            if ( next_state+1)%6 !=0: 
                                A_ = np.argmax(Q[next_state])
             
                Q[s][A] = Q[s][A] + learn_rate*(reward + discount*Q[next_state][A_] - Q[s][A])
                s = next_state
                step += 1
                R = R + reward
        
       
            RQ_List.append(R)
            no_statesQ.append(step)


            if terminal_state(s) == False:
                fails = fails +1
    
        state = state+1
    
    print(fails)
        
   
    return Q, RQ_List, no_statesQ

bestQlearningQ = np.zeros((nOS))
best_policy = np.zeros((nOS))

resultQ, RQ_List, no_statesQ = Qlearning(200000,1000, 0.1, 0.3,0.9)
print(resultQ)

print(len(RQ_List))


# the below code creates a list of lists so that reward and no of states visited
# can be tracked by state within each inner list

inner_size = 200000
newlist = [ RQ_List[i:i+inner_size] for i in range(0, len(RQ_List), inner_size) ]

newstatelist = [ no_statesQ[i:i+inner_size] for i in range(0, len(no_statesQ), inner_size) ]

final_list = []

# the below code generates an average of reward and number of states visited by state for
# each episode

for i in range(inner_size):
    num =0
    for j in range(len(newlist)):
        num = num + newlist[j][i]
    avg = num/len(newlist)
    final_list.append(avg)
 
final_state_list = []    
 
for i in range(inner_size):
    num =0
    for j in range(len(newstatelist)):
        num = num + newstatelist[j][i]
    avg = num/len(newstatelist)
    final_state_list.append(avg)
    

# the below code is used to generate optimum policy from calculated Q(s,a) values    

for s in range(nOS):
    
    bestQlearningQ[s] = max(resultQ[s])
    if terminal_state(s):
        best_policy[s] = 4
    
    elif s in wall:
        best_policy[s] = 5
        
    else:
        if s <5:
            print(s)
            
            best_policy[s]= np.argmax(resultQ[s][1:])+1
        if s ==5:
           print(s) 
           
           c = resultQ[s][2:]
           
           
           best_policy[s] = np.argmax(c)+2
            
        if s != 5:
            if (s+1)%6 ==0:
                #print (s)
                c = (resultQ[s][:1])
                cc = np.zeros((1))
                cc[0]=-999999
                
                d = (resultQ[s][2:])
                e = np.concatenate((c,cc,d))
                best_policy[s] = np.argmax(e)
        if (36-s)<=5:
            print(s)
            c = (resultQ[s][:2])
            cc = np.zeros((1))
            cc[0]=-999999
            d = (resultQ[s][3:])
            e = np.concatenate((c,cc,d))
            best_policy[s] = np.argmax(e)
        
        if s == 30:
            print (s)
            c = resultQ[s][:2]
            print(c)
            print(np.argmax(c))
            best_policy[s] = np.argmax(c)
            
        if s != 30:
            if s%6 == 0:
                print(s)
                best_policy[s] = np.argmax(resultQ[s][:3])
        if s >5:
            if s <30:
                if (s)%6 != 0:
                    if (s+1)%6 !=0: 
                        print(s)
                        best_policy[s] = np.argmax(resultQ[s])
    

for s in range(nOS):
    if s in wall:
       bestQlearningQ[s] =13   

final_result = bestQlearningQ.reshape(6,6)


print(final_result)

best_Q_policy = best_policy.reshape(6,6)
print(best_Q_policy) 


# this below code allows implementation of SARSA
# a starting state is creating from 1 to 33, and if not in wall
# process below is performed and we track the negative reward and # of states 
# visited for each episode/state


# SARSA ALGORITHM

def SARSA(no_episodes,no_steps, epsilon, learn_rate, discount):

    Q = np.zeros((nOS, nOA))
    RSARSA_List = []
    fails=0
    no_states = []
   
    
    state = 1
    
    while state < 33 and state not in wall:
        
         
        for episode in range(no_episodes):
        
        
            s= state
       
        
            step = 0
            R =0
 
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
            
            #a = random.randrange(nOA)
            
            
            while step <no_steps and terminal_state(s) != True:
                if episode ==0:
                    epsilon1 = epsilon
                else:
                    epsilon1 = epsilon/episode
                    
                A = a
                next_state, probability, reward = P[s][A]
                #print(epsilon1)
                A_ = greedy1(epsilon1, Q[next_state], episode, s)
                #A_ = np.argmax(Q[next_state])
                Q[s][A] = Q[s][A] + learn_rate*(reward + discount*Q[next_state][A_] - Q[s][A])
                s = next_state
                a = A_
           
                step += 1
                R = R + reward
    
            RSARSA_List.append(R)
            no_states.append(step)
             
        
        
            if terminal_state(s) == False:
                fails = fails +1
            
        state = state+1 
    
    print(fails)
        
    return Q, RSARSA_List, no_states

resultsarsaQ, RSARSA_List, no_states = SARSA(200000,1000, 0.1, 0.3,0.9)

# the below code creates a list of lists so that reward and no of states visited
# can be tracked by state within each inner list

inner_size = 200000
newSARSAList = [ RSARSA_List[i:i+inner_size] for i in range(0, len(RSARSA_List), inner_size) ]

newSARSAstateList = [ no_states[i:i+inner_size] for i in range(0, len(no_states), inner_size) ]

# the below code generates an average of reward and number of states visited by state for
# each episode


finalSARSA_List = []

for i in range(inner_size):
    num =0
    for j in range(len(newSARSAList)):
        num = num + newSARSAList[j][i]
    avg = num/len(newSARSAList)
    finalSARSA_List.append(avg)

finalSARSAstate_List = []

for i in range(inner_size):
    num =0
    for j in range(len(newSARSAstateList)):
        num = num + newSARSAstateList[j][i]
    avg = num/len(newSARSAstateList)
    finalSARSAstate_List.append(avg)
 
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
        bestQSARSA[s] =13
        
final_result = bestQSARSA.reshape(6,6)
print(final_result)

best_SARSA_policy = best_SARSApolicy.reshape(6,6)
print(best_SARSA_policy) 


plt.plot(RQ_List)
plt.show()
plt.plot(final_state_list)
plt.show()

plt.plot(RSARSA_List)
plt.show()        
plt.plot(finalSARSAstate_List)
plt.show()


plt.plot(no_statesQ)
plt.show()
plt.plot(no_states)
plt.show()        


# the commented code below is to just do the file creation
# for final submittal

#plt.figure(figsize = (3,3))
#plt.locator_params(axis='x', nbins=3)

#plt.xticks(fontsize =8)
#plt.yticks(fontsize =8)


#plt.xlabel("Episode No.", size = 8,)
#plt.ylabel("Cumulative Cost", size = 8)

Cum =[]

cum1 =0

for i in range(len(final_list)):
    cum1 -= final_list[i]
    Cum.append(cum1)

plt.semilogy(Cum)


Cum2 =[]

cum3 =0

for k in range(len(finalSARSA_List)):
    cum3 -= finalSARSA_List[k]
    Cum2.append(cum3)
    
plt.semilogy(Cum2)
plt.show()
#plt.legend(["Q-Learning", "SARSA"], loc ="lower right", prop = {'size': 8})
#plt.savefig('C:/Data Science and Analytics/CS 5033/line_plot1.png', bbox_inches = "tight") 


#plt.xlabel("Episode No.", size = 8,)
#plt.ylabel("No. of States Visited", size = 8)


cum1s =0
Cums = []

for j in range(len(final_state_list)):
    cum1s += final_state_list[j]
    Cums.append(cum1s)

plt.semilogy(Cums)


Cum2s = []
cum3s = 0 

for l in range(len(finalSARSAstate_List)):
    cum3s += finalSARSAstate_List[l]
    Cum2s.append(cum3s)
    

plt.semilogy(Cum2s)
plt.show()

#plt.legend(["Q-Learning", "SARSA"], loc ="lower right", prop = {'size': 8})
#plt.savefig('C:/Data Science and Analytics/CS 5033/line_plot2.png', bbox_inches = "tight") 




