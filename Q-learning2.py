import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

env = gym.make('FrozenLake-v0') #미끄러지는 부분

Q = np.zeros([env.observation_space.n, env.action_space.n])

#learning parameter 설정
learning_rate = 0.85
dis = 0.99
num_episodes = 2000

rList=[]
for i in range(num_episodes):
    #Reset environment and 첫번째 관찰
    state = env.reset()
    rAll = 0
    done = False
    # decaying E-greedy
    e = 1. / ((i // 100) + 1)
    #Q-table learning algorithm
    while not done:
        #액션의 greedy 선택
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i + 1))

        # S'의 상태와 보상
        new_state, reward, done,_ = env.step(action)

        #Update
        Q[state,action] = (1-learning_rate)*Q[state,action] + learning_rate*(reward + dis*np.max(Q[new_state,:]))

        rAll += reward
        state = new_state

    rList.append(rAll)

print("Success rate : "+str(sum(rList) / num_episodes))#승률
print("Final Q-Table Values")
print(Q)

plt.bar(range(len(rList)), rList, color="blue")
plt.show()
