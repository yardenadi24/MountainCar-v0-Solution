import gym
from gym.envs.registration import make
import numpy as np
from numpy.core.fromnumeric import argmax
from numpy.lib.function_base import gradient
from numpy.ma.core import _DomainGreaterEqual, diag
import numpy.matrixlib as matrix
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt
from collections import defaultdict
import copy
import itertools


varP = 0.04
varV = 0.0004
gamma = 1
alpha = 0.015

#action to string:
actions = ["Accelerate to the left","Don't accelerate","Accelerate to the Right"]

diagMatrix = np.array([[varP,0],[0,varV]])

def centers_generator(pos,vels):

    #pos and vel centers
    centers_p = [-0.9,-0.5,-0.1,0.2]
    centers_v = [-0.05,-0.02,-0.01,0.0,0.01,0.03,0.05,0.07]

    C = []
    #generate pairs
    for p in range(0,pos):
        for v in range(0,vels):
            C.append([centers_p[p],centers_v[v]])
        
    return np.array(C)

#generates weights
def weight_generator(pos,vels):
    W = []
    for w in range (0,3):
        W.append(np.zeros(pos*vels))
    return np.array(W)         

#generate feature vector for "state"
def feature_generator(state,centers):
    p = state[0]
    v = state[1]
    theta_features = []
    for c_i in centers:
        p_v = np.array([p,v])
        p_v_t = np.transpose(p_v)
        c_i_t = np.transpose(c_i)
        x_i = p_v_t - c_i_t
        x_i_t = np.transpose(x_i)
        invers_diag = inv(diagMatrix)
        mul1 = np.dot(x_i_t,invers_diag)
        mul2 = np.dot(mul1,x_i)
        exp = mul2 * (-0.5)
        theta_i = math.exp(exp)
        theta_features.append(theta_i)
    return np.array(theta_features)    

# Q function for feature vector, action a and Weights matrix W
def Q_func(feature_state,a,W):
    sum = 0
    for i in range(0,W[a].size):
        sum += W[a][i] * feature_state[i]
    return sum

#Epsilon greedy policy
def policy(feature_state,W,epsilon):
    prob_s = np.ones(3,dtype=float) * (epsilon/3)
    Q_s = []
    for a in range (0,3):
        Q_s.append(Q_func(feature_state,a,W)) 
    best_action = np.argmax(Q_s)
    prob_s[best_action] += (1.0-epsilon)
    return prob_s ,Q_s    

#linear Q learning algo with epsilon decay factor
def linear_Q_learning_algo(env,gamma,alpha,epsilon,epsilon_decay,num_of_pos,num_of_vels,iterBound):

    C = centers_generator(num_of_pos,num_of_vels)
    Q = Q_func
    W = weight_generator(num_of_pos,num_of_vels)
    iter = 1
    improve_stats = [[],[]]

    #episodes
    for episode in itertools.count():
        state = env.reset()
        epsilon = epsilon * epsilon_decay

        #steps for each episode
        for t in itertools.count():
            
            #prepare step
            theta_features = feature_generator(state,C)
            action_prob , Q_per_a = policy(theta_features,W,epsilon)
            action = np.random.choice([0,1,2],p=action_prob) 

            #step
            next_state,reward,done,_ = env.step(action)
            next_theta_features = feature_generator(next_state,C)
            next_action_prob ,next_Q_a = policy(next_theta_features,W,epsilon)
            next_best_action = argmax(next_Q_a)

            #if state is terminal then W = W + alpha*(R - Q(s,a,W))*gradient(Q(s,a,W))
            if done:
                W[action] = W[action] + alpha*(reward - Q_per_a[action])*theta_features

            #state isnt terminal then W = W + alpha*(R + gamma*Q(next_s,best_action,W) - Q(s,a,W))*gradient(Q(s,a,W)) 
            else:
                 W[action] = W[action] +alpha*(reward + gamma*next_Q_a[next_best_action] - Q_per_a[action])*theta_features

            #each 10k total steps calculate the policy value
            if iter % 10000 == 0 :
                print(iter)
                sim_env = copy.deepcopy(env)
                policy_val = eval_policy(sim_env,Q,W,C)
                improve_stats[0].append(iter)
                improve_stats[1].append(policy_val)  

            
          

            iter+=1
            state = next_state
            if done:
                break

        if iter > iterBound:
            break    


    return W,improve_stats        

#policy evaluation => mean of 100 simulations
def eval_policy(env_1,Q,W,C):
    sum = 0
    env_1.reset()
    print("eval")
    for iter in range(0,100):
        part_sum = simulate(env_1,Q,W,C)
        sum += part_sum
    print("finish")
    print(sum/100)
    return sum/100

#regular simulate
def simulate(env1,Q,W,C):
    state = env1.reset()
    discounter_reward = 0
    discount = 1
    for iter in itertools.count():
        features_matrix=feature_generator(state,C)
        Qval = []
        for a in range(0,3):
            Qval.append(Q(features_matrix,a,W))
        best_action = np.argmax(Qval)
        next_state, reward, done, _ = env1.step(best_action)
        discounter_reward += reward*discount
        discount = discount*gamma
        state = next_state
        if done:
            break  

    return discounter_reward

#print+render simulate
def print_simulate(env,Q,W,C):
    state = env.reset()
    total_reward = 0
    steps = 0
    for i in itertools.count():
        features_matrix=feature_generator(state,C)
        env.render()
        steps += 1
        Qval = []
        p = state[0]
        v = state[1]
        for a in range(0,3):
            Qval.append (Q(features_matrix,a,W))

        best_action = np.argmax(Qval)
        actions.append(best_action)
        next_state, reward, done, _ = env.step(best_action)
        total_reward += reward
        sign = '+' if reward > 0 else ''
        print(f'{i+1}. {p},{v} 0.5,0 {actions[best_action]} {sign}{reward}')
        state = next_state
        if done:
            break
    print(f'total steps: {steps}')
    total_reward_sign = '+' if total_reward > 0 else ''
    print(f'total rewards: {total_reward_sign}{total_reward}')


#make graph value per step
def make_Graph(stats1):
    plt.plot(stats1[0],stats1[1], label="policy value for each iteration")
    plt.title("value per step")
    plt.ylabel("value")
    plt.xlabel("step")
    plt.legend()
    plt.show()

#main
if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    W,stats = linear_Q_learning_algo(env,gamma,alpha,1.0,0.999,4,8,100000)
    print(W)
    make_Graph(stats)

    #run for test:
    C = centers_generator(4,8)
    Q= Q_func
    W = [ 
        [-24.88295967 ,-33.1727169 , -31.9668709 , -27.03305164 ,-18.75187445,
         -2.89435045  , 3.86924566  , 3.20051211 ,-25.22198471 ,-22.05481096,
  -19.90616475 ,-31.97858878 ,-48.62582375 ,-43.04460543 , -1.81572808,
   10.74256368 ,-20.91059489 ,-31.38258438 , -5.1326716  , 11.62434118,
    0.70975139 ,-45.80406691 ,-10.92055099 , 15.01222312 , -6.84906087,
  -23.1679369  ,-27.26939016 ,-30.60945199 ,-30.63299778 , -7.10217551,
   20.48692783 , 14.35627342],
 [-21.89424051 ,-20.60808263 ,-19.67197196 ,-17.81259418 ,-13.55558214,
   -2.97227012 ,  2.75933695 ,  2.73987826 ,-39.68993662 ,-21.53029368,
  -27.87397341 ,-43.41626017 ,-54.42005078 ,-35.31402133 ,  1.9549423,
   10.66260069 ,-48.98463483 ,-16.94159996  , 4.47391346 ,  3.16426773,
  -19.50800035 ,-46.56812414 , -1.7966112   ,16.78160366 ,-11.76043826,
  -19.15757126 ,-19.29904257 ,-20.34734796 ,-19.60470401 , -0.06914755,
   20.6475303  , 13.32315164],
 [-15.63197588 ,-14.00788797 ,-11.80861603 , -8.58094345 , -6.29762051,
   -6.78815333 ,  1.69937412 ,  6.74111113 ,-30.50110971 ,-28.97137866,
  -42.26804069 ,-52.74953566 ,-47.32391497 ,-16.42301325 , -5.11677537,
    8.60179914 ,-42.08786257 , -8.2673937  ,  6.1694419  , -3.2592145,
  -31.53569842 ,-51.55896237 ,  8.27053946 , 22.9413206  ,-14.21867723,
  -21.77017408 ,-18.28612348 ,-14.53131843 ,-11.57304837 ,  2.65924344,
   34.6547663  , 30.71019181]]   
    W = np.array(W)
    print_simulate(env,Q,W,C) 