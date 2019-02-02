# test of the installation of gym

import gym
import time
import random
env = gym.make('GridWorld-v0')
env.reset()
# actions = env.getAction()
actions = ['n','e','s','w']
for i in range(10):
    env.render()
    env.step(actions[int(random.random()*len(actions))]) 
    time.sleep(1)
