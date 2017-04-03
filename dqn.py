import tensorflow as tf
import os
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
    def __init__(self, motor, learning_rate):
        self.motor = motor
        self.learning_rate = learning_rate
        self.replay_memory_size = 10000
        self.experience_buffer = Memory(replay_memory_size)

    def init_network(self):
        # inputs is the observation
        self.inputs = tf.placeholder(dtype=tf.float32 , shape=[None, 363], name"observations")

        # fully connected layer
        w_fc1 = tf.Variable(tf.truncated_normal([363, 363], stddev= 0.01))
        b_fc1 = tf.Variable(tf.zeros[363])
        layer1 = tf.nn.relu(tf.matmul(inputs, w_fc1) + b_fc1)

        # Output Layer
        weight_outputs = tf.Variable(tf.truncated_normal([363, 256], stddev=0.01))
        b_out = tf.Variable(tf.zeros([256]))

        # Q value from the network
        self.Qout =  tf.matmul(layer1, weight_outputs) + b_out

        nextQ = tf.placeholder(dtype=tf.float32, shape=[None, 256])

        # loss function
        self.loss = tf.reduce_mean(tf.square(nextQ - Qout))

        optimizer = tf.train.RMSPropOptimizer(self.learning_rate)

        self.trainer = optimizer.minimize(loss)

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

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


if __name__ = "__main__":
    '''
        Define hyperparameters
    '''
    batch_size = 5
    learning_rate = 1e^-2
    gamma = 0.99
    number_of_episode = 1000
    env  = Simulator()
    
    model = dqn()
    model.init_network()
