# policy iteration (Q iteration)
import gym
import time
import random
import numpy as np
env = gym.make('GridMazeWorld-v0')
env = env.unwrapped   # BUG,坑！ ???

states = env.getStates()
actions = env.getActions()
gamma = env.getGamma()
terminal_states = env.getTerminate_states()   # ATTENSION: dictionary
v = np.zeros(len(states))   # state-value function
pi = dict()   # policy
for s in states:     # initialize policy: even policy(均匀策略)
    probality = 1/len(actions)
    for a in actions:
        key = "%d_%s"%(s,a)
        pi[key] = probality

def print_v():
    print("************************************")
    print(v[0],v[1],v[2]," ",v[3],sep='\t')
    print(v[4],v[5],v[6]," ",v[7],sep='\t')
    print(" "," ",v[8],v[9],v[10],sep='\t')
    print(v[11],v[12],v[13],v[14],v[15],sep='\t')
    print(v[16],v[17]," "," "," ",sep='\t')
    print("")

def policy_evaluation():
    for i in range(1000):
        delta = 0
        v_temp = v.copy()
        for state in states:
            if state in terminal_states:
                continue
            new_v = 0
            for action in actions:
                key = "%d_%s"%(state, action)
                r = env.getReward(state,action)
                next_state = env.transform(state,action)
                new_v +=  pi[key]*(r+gamma*v_temp[next_state-1])
            delta += abs(v[state-1]-new_v)
            v[state-1] = "%.5f"%new_v   #保留小数点后五位小数
        # if i<10:
        #     print_v()
        if delta < 1e-6:
            print("policy evaluation end before 1000 steps! current steps:",i+1)
            break

def policy_improvement():
    for state in states:
        q = dict()
        for action in actions:     # calculate q values
            q_val = 0
            r = env.getReward(state,action)
            next_state = env.transform(state,action)
            q_val = r+gamma*v[next_state-1]
            q[action] = q_val
        optimal_action = max(q,key=q.get) # BUG,注意：当有几个值同时为最大值时，返回第一个
        for action in actions:  # greedy policy/贪婪策略
            key = "%d_%s"%(state,action)
            if action == optimal_action:
                pi[key] = 1
            else:
                pi[key] = 0

def consult_pi(state): # consult the policy for which step to take in current state
    # BUG:当存在多个行动的概率一样时，取第一个
    pi_temp = dict()
    for action in actions:
        key = "%d_%s"%(state,action)
        pi_temp[action] = pi[key]
    optimal_action = max(pi_temp,key=pi_temp.get)
    return optimal_action

for i in range(10):        # training
    policy_evaluation()
    # print_v()
    policy_improvement()
    # print(pi)

################### test  ################
st = [17,1,4,6,13]
for i in range(5):
    print("round %d:"%(i+1))
    # env.reset()
    env.setState(st[i])
    env.render(mode='human')
    time.sleep(1)
    is_end = False
    step_num = 0
    while(not is_end):
        state = env.getState()
        action = consult_pi(state)
        state,r,is_end,non = env.step(action)
        step_num += 1
        env.render(mode='human')
        if is_end == True:
            print("congratulations! we arrive the terminal!")
            print("step number: %d\n"%step_num)
        time.sleep(3)

