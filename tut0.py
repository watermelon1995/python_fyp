import gym
import numpy as np

env = gym.make('FrozenLake-v0')

print env.observation_space.n
print env.action_space.n

for i in range(20):
    s = env.reset()
    for t in range(100):
        env.render();
        print ("State: {}".format(s));
        action = env.action_space.sample()
        print ("Action: {}".format(action));
        s , r , done, info = env.step(action)
        if done:
            print ("Episode finished after {} timesteps".format(t+1))
            break;
