import gym
import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
tf.disable_v2_behavior()
env = gym.make('FrozenLake-v0')

# Input and output size based on the Env
input_size = env.observation_space.n  ##16 gym환경에 setting되있음
output_size = env.action_space.n      ##4  gym환경에 이것도 setting되있음
learning_rate = 0.1

# These lines establish the feed-forward part of the network used to
# choose actions
X = tf.placeholder(shape=[1, input_size], dtype=tf.float32)  # state input
## (1,16)으로 넘긴다
##X의 행렬은 [[0,1,2,3,]] 2차원 행렬값이다
##[1*16]
W = tf.Variable(tf.random_uniform(
    [input_size, output_size], 0, 0.01))  # weight
##    입력16       출력4  Variable->학습 가능한 변수
##W의 행렬은 [[1,23,],[2,5,2,..],[2,4,5,..],[2,5,6..]]
##input*output 사이즈다
##[16*4]
Qpred = tf.matmul(X, W)  # Out Q prediction
          ## W*X 를 tensorflow에선 matmul(x,w)라 함
          ## Cost함수에서 Ws인 부분이다
Y = tf.placeholder(shape=[1, output_size], dtype=tf.float32)  # Y label
                            ## 출력 4
                            ##[ 1*16 16*4 ] = [1*4]
                            ##[0,a]를 가진다[[a1,a2,a3,a4]]
loss = tf.reduce_sum(tf.square(Y - Qpred))
        ##행렬값이므로 reduce_sum으로 하나의 값으로 합한다
        ##loss가 곧 Cost함수이다
        ##이 함수를 최소화 하면 된다
train = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate).minimize(loss)

# Set Q-learning related parameters
dis = .99
num_episodes = 2000

# Create lists to contain total rewards and steps per episode
rList = []
def one_hot(x):
    return np.identity(16)[x:x + 1]

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        # Reset environment and get first new observation
        s = env.reset()
        e = 1. / ((i / 50) + 10)
        rAll = 0
        done = False
        local_loss = []

        # The Q-Network training
        while not done:
            # Choose an action by greedily (with e chance of random action)
            # from the Q-network
            Qs = sess.run(Qpred, feed_dict={X: one_hot(s)})
            ##e-greedy방식으로 action을 선택한다
            if np.random.rand(1) < e:
                a = env.action_space.sample()
            else:
                a = np.argmax(Qs)

            # Get new state and reward from environment
            s1, reward, done, _ = env.step(a)
            #Y label을 어떻게 줄 것인가 -> 마지막인가 마지막이 아닌가에 따라 
            ## Q상태 update 방식이 달라야 함
            ## action만 update 해야하기에 Qs[0, a]로 잡아야 함
            if done:
                # Update Q, and no Qs+1, since it's a terminal state
                #Q는 2차원 배열 [[a1,a2,a3,a4]]
                Qs[0, a] = reward
            else:
                # Obtain the Q_s1 values by feeding the new state through our
                # network
                Qs1 = sess.run(Qpred, feed_dict={X: one_hot(s1)})
                #다음 상태값으로 update한다
                # Update Q
                Qs[0, a] = reward + dis * np.max(Qs1)

            # Train our network using target (Y) and predicted Q (Qpred) values
            sess.run(train, feed_dict={X: one_hot(s), Y: Qs})
            ##Qs Y레이블로 넘기고, X또한 one_hot으로 넘겨서 학습을 실행한다
            rAll += reward
            s = s1
        rList.append(rAll)

print("Percent of successful episodes: " +
      str(sum(rList) / num_episodes) + "%")
plt.bar(range(len(rList)), rList, color="blue")
plt.show()