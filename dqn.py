import tensorflow as tf
import numpy as np
import os
import math
from motor import Motor
from memory import Memory
from simulator import Simulator

"""
DQN abstraction.
As a quick reminder:
    traditional Q-learning:
        Q(s, a) += alpha * (reward(s,a) + gamma * max(Q(s') - Q(s,a))
    DQN:
        target = reward(s,a) + gamma * max(Q(s')
"""


class dqn:
    def __init__(self,  learning_rate, minibatch_size, gamma, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size
        self.replay_memory_size = 10000
        self.experience_buffer = Memory(self.replay_memory_size)

        self.init_network()

        tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        self.current_loss = 0.0


    def init_network(self):
        tf.reset_default_graph()

        '''
            Create eval q networks
        '''
        # inputs is the observation
        self.s = tf.placeholder(dtype=tf.float32 , shape=[None, self.state_space], name = "observation")

        # fully connected layer layer 1
        w_fc1 = tf.Variable(tf.truncated_normal([self.state_space, 1024], stddev= 0.01))
        b_fc1 = tf.Variable(tf.zeros([1024]))
        layer1 = tf.nn.relu(tf.matmul(self.s, w_fc1) + b_fc1)

        # fully connected layer layer 2
        w_out = tf.Variable(tf.truncated_normal([1024, self.action_space], stddev=0.01))
        b_out = tf.Variable(tf.zeros([self.action_space]))
        self.Qout =  tf.matmul(layer1, w_out) + b_out

        # q value from next state
        self.Qout_next = tf.placeholder(tf.float32, [None, self.action_space])

        '''
            Loss Function
        '''

        self.loss = tf.reduce_mean(tf.square(self.Qout_next - self.Qout))

        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)

        self.trainer = optimizer.minimize(self.loss)



        self.saver = tf.train.Saver()

        self.sess = tf.Session()


    def get_Q_values(self, state):
        return self.sess.run(self.Qout, feed_dict={self.s: [state]})[0]

    def store_experience(self, state, action, reward, nextState, done):
        self.experience_buffer.addMemory(state, action, reward, nextState, done)

    def replay_experience(self):
        if self.experience_buffer.getCurrentSize() > self.minibatch_size:
            state__miniBatch = []
            qout_miniBatch = []

            size = min(self.experience_buffer.getCurrentSize() , self.minibatch_size)
            miniBatch = self.experience_buffer.getMiniBatch(size)

            for sample in miniBatch:
                done = sample['isFinal']
                state = sample['state']
                action = sample['action']
                reward = sample['reward']
                newState = sample['newState']


                qValues = self.get_Q_values(state)

                if done:
                    qValues[action] = reward
                else:
                    qValues[action] = reward + self.gamma * np.max(self.get_Q_values(newState))

                state__miniBatch.append(state)
                qout_miniBatch.append(qValues)

            #train
            self.sess.run(self.trainer, feed_dict={self.s: state__miniBatch, self.Qout_next: qout_miniBatch})

            self.current_loss = self.sess.run(self.loss, feed_dict={self.s: state__miniBatch, self.Qout_next: qout_miniBatch})

    def load_model(self, model_path=None):
        if model_path:
            # load from saved file
            self.saver.restore(self.sess, model_path)
        else:
            # load from checkpoint
            checkpoint = tf.train.get_checkpoint_state('/home/kin/python/q_learning/checkpoint')
            if checkpoint and checkpoint.model_checkpoint_path:
                self.saver.restore (self.sess, checkpoint.model_checkpoint_path)

    def save_model(self):
        self.saver.save(self.sess, '/home/kin/python/q_learning/saved_model.ckpt')


if __name__ == "__main__":
    '''
        Define hyperparameters
    '''
    learning_rate = 0.01
    gamma = 0.99
    explore_step = 2000
    initial_epsilon = 1.0
    final_epsilon = 0.0001
    state_space = 363
    action_space = 256

    minibatch_size = 16
    learnStart = 16

    epsilon_discount = 0.9986
    env  = Simulator()
    model = dqn(learning_rate, minibatch_size, gamma, state_space , action_space)

    counter = 0
    highest_reward = 0
    epsilon = initial_epsilon
    last_info = ""

    for x in range(explore_step):
        observation = env.reset()
        done = False
        cumulated_reward = 0
        counter = 0
        loss = 0.0
        Q_max = 0.0
        while not done:
            # Choose Action
            rand = np.random.random()
            if rand < epsilon:
                print "----------------Random Action-------------------"
                action = np.random.randint(low = 0, high = len(env.motor.action_space))
            else:
                action = np.argmax(model.get_Q_values(observation))

            newObservation, reward , done ,debug = env.step(action)

            cumulated_reward += reward
            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            model.store_experience(observation, action, reward, newObservation, done)

            model.replay_experience()


            loss += model.current_loss
            last_info = debug
            Q_max += np.max(model.get_Q_values(observation))
            counter += 1
            observation = newObservation

        print "Epsode finished at {}, status: {} , reward : {}, loss: {:.4f}, Q_max : {} ".format(counter, last_info, cumulated_reward, loss/counter, Q_max/counter)
        epsilon *= epsilon_discount

        if x % 100 == 0 and x != 0:
            model.save_model()
