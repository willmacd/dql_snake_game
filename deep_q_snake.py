""" Child class of Snake to be used for Deep Q-Learning implementation of Snake game """
import numpy as np
import tensorflow as tf

from snake import Snake


class Deep_Q_Snake(Snake):
    def __init__(self, block_size: int = 20):
        """
        Constructor for Deep Q Learning Snake; defining all functionality of the Deep Q Network child of the base game
        Snake class

        :param block_size: Size of single square within the game board
        """
        super(Deep_Q_Snake, self).__init__(block_size=block_size)

        # Define the rewards for specified actions
        self.food_reward = 10
        self.collision_reward = -10
        self.else_reward = 0

        # Define the exploration rate - TODO: Try exponential decay of epsilon with as episodes increase
        self.__epsilon = float(1 / 75)

        # Define discount factor
        self.__gamma = 1

        self.__states_replay_buffer = np.array([]).reshape(0, 11)
        self.__rewards_replay_buffer = np.array([]).reshape(0, 1)

        # Define Deep Q-Learning model architecture
        # Initialize a sequential neural network architecture and its input layer
        self.__dql_model = tf.keras.Sequential()
        self.__dql_model.add(tf.keras.layers.InputLayer(input_shape=(11, )))
        # Initialize the first fully connected layer followed by ReLU activation and Dropout
        self.__dql_model.add(tf.keras.layers.Dense(100, activation='relu'))
        self.__dql_model.add(tf.keras.layers.Dropout(0.3))
        # Initialize the second fully connected layer followed by ReLU activation and Dropout
        self.__dql_model.add(tf.keras.layers.Dense(100, activation='relu'))
        self.__dql_model.add(tf.keras.layers.Dropout(0.3))
        # Initialize the third fully connected layer followed by ReLU activation and Dropout
        self.__dql_model.add(tf.keras.layers.Dense(100, activation='relu'))
        self.__dql_model.add(tf.keras.layers.Dropout(0.3))
        # Initialize the output fully connected layer followed by Softmax activation
        self.__dql_model.add(tf.keras.layers.Dense(3, activation='relu'))

    def __epsilon_greedy_selection(self, state_vector):
        """
        Determine the action to be taken based on the epsilon-greedy selection method

        :param state_vector: State vector describing the Snake's observable current observable environment
        :return:
        """
        state_vector = state_vector.reshape(1, -1)
        if np.random.uniform(0, 1) < self.__epsilon:
            action = np.random.choice([0, 1, 2])
        else:
            action = np.argmax(self.state_action_q_values(state_vector=state_vector))
        return action

    def dql_update(self, food_loc, window_width, window_height):
        """
        Update the trajectory of the snake at a given point in time based on the determined action from the DQL network

        :param food_loc: Location of the food within the game screen
        :param window_width: Width of the game window in which the snake can move
        :param window_height: Height of the game window in which the snake can move
        :return:
        """
        # Select an action to be taken based on the Epsilon-Greedy selection method
        action = self.__epsilon_greedy_selection(state_vector=self.state_vector(food_loc=food_loc,
                                                                                window_width=window_width,
                                                                                window_height=window_height))

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

    def state_vector(self, food_loc, window_width, window_height):
        """
        Function to generate the observable state vector for the snake at any given point in time

        :param food_loc: Location of the food within the game screen - to be converted to binary values such as:
        'food_left', 'food_right', 'food_up', and 'food_down'
        :param window_width: Width of the game window in which the snake can move
        :param window_height: Height of the game window in which the snake can move
        :return:
        """
        # Initialize the state vector list
        state_vector = []

        # Initialize binary variables for `danger_straight`, `danger_left` and `danger_right`
        danger_straight = 0
        danger_left = 0
        danger_right = 0

        # If Snake's location is against the leftmost border
        if self.rect.x == self.block_size:
            # And Snake's trajectory is moving left
            if self.trajectory.x < 0:
                # Snake is moving directly toward the border, set `danger_straight` to 1
                danger_straight = 1
        # If Snake's location is against the rightmost border (one away form border since triggering occurs upon
        # entering the next state)
        elif self.rect.x == (window_width - self.block_size - self.block_size):
            # And Snake's trajectory is moving right
            if self.trajectory.x > 0:
                # Snake is moving directly toward the border, set `danger_straight` to 1
                danger_straight = 1

        # If Snake's location is against the topmost border
        if self.rect.y == self.block_size:
            # And Snake's trajectory is moving up
            if self.trajectory.y < 0:
                # Snake is moving directly toward the border, set `danger_straight` to 1
                danger_straight = 1
        # If Snake's location is against the bottommost border (one away form border since triggering occurs upon
        # entering the next state)
        elif self.rect.y == (window_height - self.block_size - self.block_size):
            # And Snake's trajectory is moving down
            if self.trajectory.y > 0:
                # Snake is moving directly toward the border, set `danger_straight` to 1
                danger_straight = 1

        # If Snake's location is against the leftmost border
        if self.rect.x == 0:
            # And Snake's trajectory is moving up
            if self.trajectory.y < 0:
                # Snake is moving up along leftmost border, set `danger_left` to 1
                danger_left = 1
            # And Snake's trajectory is moving down
            elif self.trajectory.y > 0:
                # Snake is moving down along the leftmost border, set `danger_right` to 1
                danger_right = 1
        # If Snake's location is against the rightmost border
        elif self.rect.x == (window_width - self.block_size):
            # And Snake's trajectory is moving up
            if self.trajectory.y < 0:
                # Snake is moving up along the rightmost border, set `danger_right` to 1
                danger_right = 1
            # And Snake's trajectory is moving down
            elif self.trajectory.y > 0:
                # Snake is moving down along the rightmost border, set `danger_left` to 1
                danger_left = 1

        if self.rect.y == 0:
            # And Snake's trajectory is moving left
            if self.trajectory.x < 0:
                # Snake is moving left along the topmost border, set `danger_right` to 1
                danger_right = 1
            # And Snake's trajectory is moving right
            elif self.trajectory.x > 0:
                # Snake is moving right along the topmost border, set `danger_left` to 1
                danger_left = 1
        # If Snake's location is against the bottommost border
        elif self.rect.y == (window_height - self.block_size):
            # And Snake's trajectory is moving left
            if self.trajectory.x < 0:
                # Snake is moving left along the bottommost border, set `danger_left` to 1
                danger_left = 1
            # And Snake's trajectory is moving right
            elif self.trajectory.x > 0:
                # Snake is moving right along the bottommost border, set `danger_right` to 1
                danger_right = 1

        # Check all of the cells in the Snake's body
        for cell in self.entire_snake[:-1]:
            # If a body cell is located one cell to the right of the Snake's head
            if cell == [self.rect.x + self.block_size + self.block_size, self.rect.y]:
                # And Snake's trajectory is moving right
                if self.trajectory.x > 0:
                    # Set `danger_straight` to 1
                    danger_straight = 1

            # If a body cell is located one cell to the left of the Snake's head
            if cell == [self.rect.x - self.block_size - self.block_size, self.rect.y]:
                # And Snake's trajectory is moving left
                if self.trajectory.x < 0:
                    # Set `danger_straight` to 1
                    danger_straight = 1

            # If a body cell is located one cell below the Snake's head
            if cell == [self.rect.x, self.rect.y + self.block_size + self.block_size]:
                # And Snake's trajectory is moving left
                if self.trajectory.x < 0:
                    # Set `danger_left` to 1
                    danger_left = 1

            # If a body cell is located one cell above the Snake's head
            if cell == [self.rect.x, self.rect.y - self.block_size - self.block_size]:
                # And Snake's trajectory is moving up
                if self.trajectory.y < 0:
                    # Set `danger_straight` to 1
                    danger_straight = 1

            # If a body cell is located one cell to the right of the Snake's head
            if cell == [self.rect.x + self.block_size, self.rect.y]:
                # And Snake's trajectory is moving up
                if self.trajectory.y < 0:
                    # Set `danger_right` to 1
                    danger_right = 1
                # And Snake's trajectory is moving down
                elif self.trajectory.y > 0:
                    # Set `danger_left` to 1
                    danger_left = 1

            # If a body cell is located one cell to the left of the Snake's head
            if cell == [self.rect.x - self.block_size, self.rect.y]:
                # And Snake's trajectory is moving up
                if self.trajectory.y < 0:
                    # Set `danger_left` to 1
                    danger_left = 1
                # And Snake's trajectory is moving down
                elif self.trajectory.y > 0:
                    # Set `danger_right` to 1
                    danger_right = 1

            # If a body cell is located one cell below the Snake's head
            if cell == [self.rect.x, self.rect.y + self.block_size]:
                # And Snake's trajectory is moving left
                if self.trajectory.x < 0:
                    # Set `danger_left` to 1
                    danger_left = 1
                # And Snake's trajectory is moving right
                elif self.trajectory.x > 0:
                    # Set `danger_right` to 1
                    danger_right = 1

            # If a body cell is located one cell above the Snake's head
            if cell == [self.rect.x, self.rect.y - self.block_size]:
                # And Snake's trajectory is moving left
                if self.trajectory.x < 0:
                    # Set `danger_right` to 1
                    danger_right = 1
                # And Snake's trajectory is moving right
                elif self.trajectory.x > 0:
                    # Set `danger_left` to 1
                    danger_left = 1

        state_vector.append(danger_straight)
        state_vector.append(danger_left)
        state_vector.append(danger_right)

        # Set `moving_left` to 1 if the Snake's X trajectory is less than 0 (i.e., -20); otherwise, set to 0
        moving_left = 1 if (self.trajectory.x < 0 and self.trajectory.y == 0) else 0
        state_vector.append(moving_left)

        # Set `moving_right` to 1 if the Snake's X trajectory is greater than 0 (i.e., 20); otherwise, set to 0
        moving_right = 1 if (self.trajectory.x > 0 and self.trajectory.y == 0) else 0
        state_vector.append(moving_right)

        # Set `moving_up` to 1 if the Snake's Y trajectory is less than 0 (i.e., -20); otherwise, set to 0
        moving_up = 1 if (self.trajectory.x == 0 and self.trajectory.y < 0) else 0
        state_vector.append(moving_up)

        # Set `moving_down` to 1 if the Snake's Y trajectory is greater than 0 (i.e., 20); otherwise, set to 0
        moving_down = 1 if (self.trajectory.x == 0 and self.trajectory.y > 0) else 0
        state_vector.append(moving_down)

        # Set `food_left` to 1 if the X location of the Snake is greater than that of the food; otherwise, set to 0
        food_left = 1 if self.rect.x > food_loc.x else 0
        state_vector.append(food_left)

        # Set `food_right` to 1 if the X location of the Snake is less than that of the food; otherwise, set to 0
        food_right = 1 if self.rect.x < food_loc.x else 0
        state_vector.append(food_right)

        # Set `food_up` to 1 if the Y location of the Snake is greater than that of the food; otherwise, set to 0
        food_up = 1 if self.rect.y > food_loc.y else 0
        state_vector.append(food_up)

        # Set `food_down` to 1 if the Y location of the Snake is less than that of the food; otherwise, set to0
        food_down = 1 if self.rect.y < food_loc.y else 0
        state_vector.append(food_down)

        # Convert to `numpy.ndarray` for passing into the neural network
        state_vector = np.array(state_vector)
        return state_vector

    def update_replay_buffer(self, state_vector: list, reward: int):
        """
        Update the `np.ndarray` containing tuples of state vectors and their respective rewards

        :param state_vector: State vector for current state at which the agent is located
        :param reward: Reward value for entering the current state
        :return:
        """
        self.__states_replay_buffer = np.vstack((self.__states_replay_buffer, state_vector))
        self.__rewards_replay_buffer = np.vstack((self.__rewards_replay_buffer, reward))

    def summarize_dql_network(self):
        """
        Display Deep Q-Learning model architecture

        :return:
        """
        self.__dql_model.summary()

    def compile(self, learning_rate: float = 0.00005):
        """
        Compile Deep Q-Learning model to instantiate optimizer, loss function and metrics for use in the training
        process of the model

        :param learning_rate: Learning rate value to be used in the model optimizer function
        :return:
        """
        self.__dql_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                                 loss=tf.keras.losses.MeanSquaredError(),
                                 metrics=['accuracy'])

    def state_action_q_values(self, state_vector):
        """
        Function to return the Q values of each action given the input state vector

        :param state_vector: State vector describing the Snake's observable current observable environment
        :return:
        """
        return self.__dql_model.predict(x=state_vector)

    def train(self, epochs: int = 1, verbose: int = 0):
        """
        Through playing the game and collecting samples of actions

        :param epochs:
        :param verbose:
        :return:
        """
        assert(verbose in [0, 1])

        target_Qs = np.array([]).reshape(0, 3)
        for state, reward in zip(self.__states_replay_buffer, self.__rewards_replay_buffer):
            if state.shape[0] != reward.shape[0]:
                state = state.reshape(1, -1)
            Q_vals = self.state_action_q_values(state)[0]
            target = (reward + self.__gamma * np.amax(Q_vals))
            target_q_val = self.state_action_q_values(state)
            target_q_val[0][np.argmax(self.state_action_q_values(state))] = target
            target_Qs = np.vstack((target_Qs, target_q_val))
            # self.__dql_model.fit(x=state, y=target_q_val)

        self.__dql_model.fit(x=self.__states_replay_buffer, y=target_Qs, batch_size=800, epochs=epochs, verbose=verbose)

        self.__states_replay_buffer = np.array([]).reshape(0, 11)
        self.__rewards_replay_buffer = np.array([]).reshape(0, 1)
