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
        self.speed = 10
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

        # Initialize variables to track locations of snake components
        head = []

        # Append the current locations of the snakes components to their respective lists
        head.append(self.rect.x)
        head.append(self.rect.y)
        self.entire_snake.append(head)

        # If the length of the entire_snake array is larger than the `self.snake_length` array, then remove the first
        # set of coordinates since the snake has moved out of this location
        if len(self.entire_snake) > self.snake_length:
            del self.entire_snake[0]

        collision = self.__check_collision(snake_head_coordinates=head, snake_coordinates_list=self.entire_snake)

        return collision

    def __check_collision(self, snake_head_coordinates: list, snake_coordinates_list: list):
        """
        Check whether the snake's head has collided with any part of its tail

        :param snake_head_coordinates: List of coordinates for where the head of the snake is located
        :param snake_coordinates_list: List of coordinates for each component of the snakes body
        :return:
        """
        # Loop through each cell of the snakes body and determine if the head of the snake has made a collision with the
        # any section of the snakes tail
        collision = False
        for cell in snake_coordinates_list[:-1]:
            # If so, indicate that a collision has occurred and the game should end
            if cell == snake_head_coordinates:
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

    def state_vector(self, food_loc):
        """
        Function to generate the observable state vector for the snake at any given point in time

        :param food_loc: Location of the food within the game screen - to be converted to binary values such as:
        'food_left', 'food_right', 'food_up', and 'food_down'
        :return:
        """
        # Initialize the state vector list
        state_vector = []

        # TODO: Define boolean value to indicate whether there is a threat immediately in front of Snake's head
        danger_straight = None
        state_vector.append(danger_straight)

        # TODO: Define boolean value to indicate whether there is a threat immediately to the left of the Snake's head
        danger_left = None
        state_vector.append(danger_left)

        # TODO: Define boolean value to indicate whether there is a thread immediately to the right of the Snake's head
        danger_right = None
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
