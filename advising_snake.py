""" Child class of Snake to be used for Action Advising implementation of Snake game """
import os
from random import random

import numpy as np
import tensorflow as tf
import random
from deep_q_snake import Deep_Q_Snake

from snake import Snake


class Advising_Snake(Snake):
    def __init__(self, block_size: int = 20):
        """
        Constructor for Deep Q Learning Snake; defining all functionality of the Deep Q Network child of the base game
        Snake class

        :param block_size: Size of single square within the game board
        """
        super(Advising_Snake, self).__init__(block_size=block_size)

        # Ther will be one head for each DQN output
        self.head_number = 3

        # Override the snake speed attribute to make training process significantly faster
        self.speed = 20

        # Define the rewards for specified actions
        self.food_reward = 10.
        self.collision_reward = -10.
        self.else_reward = 0.

        # Define the exploration rate
        self.__epsilon = float(1 / 75)

        # Define discount factor
        self.__gamma = 0.75

        self.__state_replay_buffer = np.array([]).reshape(0, 11)
        self.__action_replay_buffer = np.array([]).reshape(0, 1)
        self.__reward_replay_buffer = np.array([]).reshape(0, 1)
        self.__state_prime_replay_buffer = np.array([]).reshape(0, 11)
        self.__terminal_replay_buffer = np.array([]).reshape(0, 1)

        #self.mini_sprime = []

        #TODO change here for mutliple heads and minibatch entry
        # Define Deep Q-Learning model architecture
        # Initialize a sequential neural network architecture and its input layer
        #self.__dql_model = tf.keras.Sequential()
        #self.__dql_model.add(tf.keras.layers.InputLayer(input_shape=(11, )))
        # Initialize the first fully connected layer followed by ReLU activation and Dropout
        # This is the backbone
        #self.__dql_model.add(tf.keras.layers.Dense(100, activation='relu'))
        # self.__dql_model.add(tf.keras.layers.Dropout(0.3))
        # Initialize the second fully connected layer followed by ReLU activation and Dropout
        #self.__dql_model.add(tf.keras.layers.Dense(100, activation='relu'))
        # self.__dql_model.add(tf.keras.layers.Dropout(0.3))
        # Initialize the third fully connected layer followed by ReLU activation and Dropout
        #self.__dql_model.add(tf.keras.layers.Dense(100, activation='relu'))
        # self.__dql_model.add(tf.keras.layers.Dropout(0.3))
        # Initialize the output fully connected layer followed by Softmax activation
        #self.__dql_model.add(tf.keras.layers.Dense(3, activation='softmax'))

        input = tf.keras.layers.Input(shape = (11, ))
        layer1 = tf.keras.layers.Dense(100, activation='relu')(input)
        layer2 = tf.keras.layers.Dense(100, activation='relu')(layer1)
        layer3 = tf.keras.layers.Dense(100, activation='relu')(layer2)
        head1 = tf.keras.layers.Dense(3, activation='softmax')(layer3)
        head2 = tf.keras.layers.Dense(3, activation='softmax')(layer3)
        head3 = tf.keras.layers.Dense(3, activation='softmax')(layer3)

        self.__dql_model = tf.keras.models.Model(inputs=input, outputs=[head1, head2, head2])

    def minibatch(self, batch_size):
        """
        Minibatch for the update

        :param: batch size
        :return: encoded action
        """
        #add len here
        samples = np.floor(np.random.random((self.__state_replay_buffer,)) * batch_size)

        # select the experience from the sampled index
        self.mini_state = [self.__state_replay_buffer[int(i)] for i in samples]
        self.mini_reward = [self.__reward_replay_buffer[int(i)] for i in samples]
        self.mini_action = [self.__action_replay_buffer[int(i)] for i in samples]
        self.mini_sprime = [self.__state_prime_replay_buffer[int(i)] for i in samples]
        self.mini_terminal = [self.__terminal_replay_buffer[int(i)] for i in samples]

        return self.mini_state, self.mini_reward, self.mini_action, self.mini_sprime, self.mini_terminal

    
    def one_hot_encoding(self, action):
        """
        For one hot encoding the actions 

        :param: action
        :return: encoded action
        """
        indices = [0, 1, 2]
        depth = len(action)
        action = tf.one_hot(indices, depth)

        return action
    
    def target_network(self):

        target_Q = tf.argmax(Deep_Q_Snake.state_action_q_values(self.mini_sprime))

        return target_Q

    def availability(percent=75):
        return random.randrange(100) < percent

    #TODO: fix
    def sample_selection(self):
        """
        For updating each head for different samples to reduce bias. Samples |D|h numbers between 
        {0, 1} with fixed probability

        :return: minibatches of replay buffers except terminal
        """
        # samples = np.random.choice(len(self.__state_replay_buffer), batch_size, replace=False)
        # Rnadomly sampling some batches from experience bugger
        samples = np.floor(np.random.random((self.__state_replay_buffer,))*self.head_number)

        # select the experience from the sampled index
        mini_state = [self.__state_replay_buffer[int(i)] for i in samples]
        mini_reward = [self.__reward_replay_buffer[int(i)] for i in samples]
        mini_action = [self.__action_replay_buffer[int(i)] for i in samples]
       # mini_sprime = [self.__state_prime_replay_buffer[int(i)] for i in samples]
        mini_terminal = [self.__terminal_replay_buffer[int(i)] for i in samples]

        return mini_state, mini_reward, mini_action, mini_sprime, mini_terminal
    #TODO: fill this
    def uncertainty_estimator(self, state_vector):

        pass

