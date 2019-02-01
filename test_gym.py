# test of the installation of gym

import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
    env.render() # update the output (image)/图像引擎
    env.step(env.action_space.sample()) # take a random action/物理引擎 
