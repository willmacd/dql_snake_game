""" Child class of Snake to be used for Action Advising implementation of Snake game """
import os
from random import random
from statistics import variance

import numpy as np
import tensorflow as tf
import random
from deep_q_snake import Deep_Q_Snake
from snake import Snake


class Advising_Snake(Deep_Q_Snake):
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
        self.variance = 0

        # Define discount factor
        self.__gamma = 0.75

        self.__state_replay_buffer = np.array([]).reshape(0, 11)
        self.__action_replay_buffer = np.array([]).reshape(0, 1)
        self.__reward_replay_buffer = np.array([]).reshape(0, 1)
        self.__state_prime_replay_buffer = np.array([]).reshape(0, 11)
        self.__terminal_replay_buffer = np.array([]).reshape(0, 1)

        #self.mini_sprime = []

        self.input = tf.keras.layers.Input(shape = (11, ))
        self.layer1 = tf.keras.layers.Dense(100, activation='relu')(input)
        self.layer2 = tf.keras.layers.Dense(100, activation='relu')(self.layer1)
        self.layer3 = tf.keras.layers.Dense(100, activation='relu')(self.layer2)
        self.head1 = tf.keras.layers.Dense(3, activation='softmax')(self.layer3)
        self.head2 = tf.keras.layers.Dense(3, activation='softmax')(self.layer3)
        self.head3 = tf.keras.layers.Dense(3, activation='softmax')(self.layer3)

        self.__dql_model = tf.keras.models.Model(inputs=input, outputs=[self.head1, self.head2, self.head3])

    def minibatch(self, batch_size):
        """
        Minibatch for the update

        :param: batch size
        :return: 
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
        :return: 
        """
        indices = [0, 1, 2]
        depth = len(action)
        action = tf.one_hot(indices, depth)

        return action
    
    def target_network(self, mini_sprime):
        """
        Calculate the Q value with the minibatch state prime and return the max of all actions

        :param minibatch of state prime
        :return:
        """

        target_Q = np.amax(self.state_action_q_values(mini_sprime), axis = 0)

        return target_Q

    def availability(self, percent=75):
        """
        Randomly assign True or False for agent's availability. Returns true 75% of the time, meaning 
        the agent is available 75% of the time

        :param percentage it would return True
        :return:
        """
        
        return random.randrange(100) < percent

    def advising_update(self, action):
        """
        Update the trajectory of the snake at a given point in time based on the determined action from the DQL network

        :param food_loc: Location of the food within the game screen
        :param window_width: Width of the game window in which the snake can move
        :param window_height: Height of the game window in which the snake can move
        :return:
        """
    
        # If selected action is to turn left (action 1)
        if action == 1:
            # And current trajectory is moving West
            if self.trajectory.x < 0:
                # Change trajectory to be moving South
                self.trajectory.x = 0
                self.trajectory.y = self.block_size
            # And current trajectory is moving East
            elif self.trajectory.x > 0:
                # Change trajectory to be moving North
                self.trajectory.x = 0
                self.trajectory.y = -self.block_size
            # And current trajectory is moving North
            elif self.trajectory.y < 0:
                # Change trajectory to be moving West
                self.trajectory.x = -self.block_size
                self.trajectory.y = 0
            # And current trajectory is moving South
            elif self.trajectory.y > 0:
                # Change trajectory to be moving East
                self.trajectory.x = self.block_size
                self.trajectory.y = 0
        # If selected action is to turn right (action 2)
        elif action == 2:
            # And current trajectory is moving West
            if self.trajectory.x < 0:
                # Change trajectory to be moving North
                self.trajectory.x = 0
                self.trajectory.y = -self.block_size
            # And current trajectory is moving East
            elif self.trajectory.x > 0:
                # Change trajectory to be moving South
                self.trajectory.x = 0
                self.trajectory.y = self.block_size
            # And current trajectory is moving North
            elif self.trajectory.y < 0:
                # Change trajectory to be moving East
                self.trajectory.x = self.block_size
                self.trajectory.y = 0
            # And current trajectory is moving South
            elif self.trajectory.y > 0:
                # Change trajectory to be moving West
                self.trajectory.x = -self.block_size
                self.trajectory.y = 0
        # Otherwise, selected action is to continue straight (action 0)
        else:
            # Continue with the current specified trajectory
            self.trajectory = self.trajectory
        return action

    def uncertainty_estimator(self, state_vector):
        """
        Estimates uncertinty for asking for advice. Calculates the variance of Q values for each head.
        and divided by the number of actions

        :param state_vector: for calculating the Q values
        :return:
        """
        for i in range(self.head_number):
            #self.__dql_model.outputs[i]
            self.variance += np.var(self.state_action_q_values(state_vector)[i])
            self.uncertainty = self.variance/3

            return self.uncertainty
        
    def train(self, epochs: int = 1, verbose: int = 0):
        """
        Through playing the game and collecting samples of actions

        :param epochs: Number of epochs of training to subject the network to for each episode
        :param verbose: Boolean flag indicating whether or not to display information about training
        :return:
        """
        # Assert that the input values for verbosity matches that which is expected
        assert(verbose in [0, 1])

        # Initialize an empty array of length 3 to store Q value approximations for each state vector
        target_Qs = np.array([]).reshape(0, 3)

        # Loop through each of the replay buffers values (zipped together to preserve order)
        for state, reward, action, state_prime, terminal in zip(self.minibatch( self.__state_replay_buffer,
                                                                                self.__reward_replay_buffer,
                                                                                self.__action_replay_buffer,
                                                                                self.__state_prime_replay_buffer,
                                                                                self.__terminal_replay_buffer)):
            # Ensure that the state vector is in the correct shape to be passed into the network
            if state.shape[0] != reward.shape[0]:
                state = state.reshape(1, -1)
            # Ensure that the state prime vector is in the correct shape to be passed into the network
            if state_prime.shape[0] != reward.shape[0]:
                state_prime = state_prime.reshape(1, -1)

            #One hot encodes the action coming from minibatch
            # This might go
            action = self.one_hot_encoding(action)

            # Get the max Q value for the state from all actions 
            target_Qs = self.target_network(state_prime)

            # Utilize the Deep Q Network as the Q-value policy to predict action to take at current state
            state_Q_vals = self.state_action_q_values(state)[0]

            for i in range(self.head_number):

                if not terminal:
                    
                    #average should be a list 1x3 of the averaged actions
                    average = tf.keras.layers.Average()([self.head1, self.head2, self.head3])

                    # Utilize the Deep Q Network as the Q-value policy to predict action to take at next state
                    state_prime_Q_vals = self.state_action_q_values(state_prime)[0]

                    # Calculate the Q value update from the action selected for the next state
                    target_Q_val = (reward + (self.__gamma * np.max(average)))
                else:
                    target_Q_val = reward

                # Update the Q val for the action taken at current time step to be the calculated Q value update
                state_Q_vals[int(action)] = target_Q_val
                state_Q_vals = state_Q_vals.reshape(1, -1)

            if verbose == 1:
                print(state)
                print(state_Q_vals)

            # Train the Deep Q Network on the current observable state and the target Q values
            self.__dql_model.fit(x=state, y=state_Q_vals, epochs=epochs,
                                 verbose=verbose)

        # Reset all of the replay buffers at the end of each episodes training stage
        self.__state_replay_buffer = np.array([]).reshape(0, 11)
        self.__action_replay_buffer = np.array([]).reshape(0, 1)
        self.__reward_replay_buffer = np.array([]).reshape(0, 1)
        self.__state_prime_replay_buffer = np.array([]).reshape(0, 11)
        self.__terminal_replay_buffer = np.array([]).reshape(0, 1)
