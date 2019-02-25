# test of the Maze environment

import gym
import time
import random
env = gym.make('GridMazeWorld-v0')
env = env.unwrapped

env.reset()

actions = ['n','e','s','w']
for i in range(20):
    env.render(mode='human')
    env.step(actions[int(random.random()*len(actions))]) 
    time.sleep(1)
