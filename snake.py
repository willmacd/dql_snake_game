""" Class file containing necessary functionality for the Snake asset within the Snake game """
import pygame

import numpy as np
import tensorflow as tf


class Snake(pygame.sprite.Sprite):
    def __init__(self, block_size: int = 20):
        """
        Constructor for Snake sprite class; defining all functionality of the Snake asset in the snake game

        :param block_size: Size of single square within the game board
        """
        super(Snake, self).__init__()
        # Initialize game surface for agent, and assign agent a color
        self.surf = pygame.Surface((block_size, block_size))
        self.surf.fill((0, 0, 139))

        # Define attribute to define the size of game board cells
        self.block_size = block_size

        # Define agent head within game surface
        self.rect = self.surf.get_rect()

        # Define an attribute to contain the overall length of the snake and a list of coordinates for each cell the
        # snake encompasses
        self.snake_length = 1
        self.entire_snake = []

        # Define trajectory of snake agents head and the speed at which it travels (one block per time-step)
        self.speed = 5
        self.trajectory = pygame.math.Vector2(block_size, 0)

        # Define the rewards for specified actions
        self.food_reward = 10
        self.collision_reward = -10
        self.else_reward = 0

        # Define the exploration rate
        self.exploration_rate = float(1/75)

        # Define Deep Q-Learning model architecture
        # Initialize a sequential neural network architecture and its input layer
        self.__dql_model = tf.keras.Sequential()
        self.__dql_model.add(tf.keras.layers.InputLayer(input_shape=11))
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

    def update(self, event):
        """
        Update the trajectory of the snake at a given point in time based on the determined action from the DQL network

        :param event: pygame event that occurred at current time-step
        :return:
        """
        # Base Snake game functionality - To be replaced by DQL agent policy
        if (event.key == pygame.K_LEFT or event.key == pygame.K_a) and self.trajectory.x == 0:
            self.trajectory.x = -self.block_size
            self.trajectory.y = 0
        elif (event.key == pygame.K_RIGHT or event.key == pygame.K_d) and self.trajectory.x == 0:
            self.trajectory.x = self.block_size
            self.trajectory.y = 0
        elif (event.key == pygame.K_UP or event.key == pygame.K_w) and self.trajectory.y == 0:
            self.trajectory.x = 0
            self.trajectory.y = -self.block_size
        elif (event.key == pygame.K_DOWN or event.key == pygame.K_s) and self.trajectory.y == 0:
            self.trajectory.x = 0
            self.trajectory.y = self.block_size

    def move_snake(self):
        """
        Move snake within game screen according to the snakes current trajectory

        :return:
        """
        # Update the current location of the snake based on the
        self.rect.x += self.trajectory.x
        self.rect.y += self.trajectory.y

        # Append the current locations of the snakes components to their respective lists
        self.entire_snake.append([self.rect.x, self.rect.y])

        # If the length of the entire_snake array is larger than the `self.snake_length` array, then remove the first
        # set of coordinates since the snake has moved out of this location
        if len(self.entire_snake) > self.snake_length:
            del self.entire_snake[0]

        collision = self.__check_collision()

        return collision

    def __check_collision(self):
        """
        Check whether the snake's head has collided with any part of its tail

        :return:
        """
        # Loop through each cell of the snakes body and determine if the head of the snake has made a collision with the
        # any section of the snakes tail
        collision = False
        for cell in self.entire_snake[:-1]:
            # If so, indicate that a collision has occurred and the game should end
            if cell == [self.rect.x, self.rect.y]:
                collision = True
        return collision

    def display_snake(self, game_screen):
        """
        Function for displaying all components of the snake within the game screen

        :param game_screen: Reference to window in which the game is being hosted
        :return:
        """
        for x in self.entire_snake:
            pygame.draw.rect(game_screen, (0, 0, 139), [x[0], x[1], self.block_size, self.block_size])

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

    def summarize_dql_network(self):
        """
        Display Deep Q-Learning model architecture

        :return:
        """
        self.__dql_model.summary()

    def __compile(self, learning_rate: float = 0.00005):
        """
        Compile Deep Q-Learning model to instantiate optimizer, loss function and metrics for use in the training
        process of the model

        :param learning_rate: Learning rate value to be used in the model optimizer function
        :return:
        """
        self.__dql_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                                 loss=tf.keras.losses.MeanSquaredError(),
                                 metrics=['accuracy'])
