""" Main file for Action Advising for Snake implementation """
import sys
import math
import time

import numpy as np
import pygame

from deep_q_snake import Deep_Q_Snake
from advising_snake import Advising_Snake
from food import FoodPellet

# Initialize global game variables
BACKGROUND = (0, 0, 0)
BORDER = (200, 200, 200)
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400
BLOCK_SIZE = 20

global SCREEN, CLOCK

# Initialize global hyper-parameters
EPISODES = 500
MAX_STEPS_SINCE_FOOD = 200
LR = 0.000075


if __name__ == '__main__':
    # Initialize an instance of pygame for development
    pygame.init()

    # Instantiate a window containing the game board and define a game clock
    SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    SCREEN.fill(BACKGROUND)
    CLOCK = pygame.time.Clock()

    snake = Advising_Snake()
    snake.load_weights("./saved_models/initial_train/dql_snake.h5")
    snake.compile(learning_rate=LR)

    # Update display and set a Window name for the game
    pygame.display.update()
    pygame.display.set_caption("Action Advising DQN Snake Game")

    best_score = 0
    spin_out_count = 0
    for episode in range(1, EPISODES+1):
        print("Episode {}".format(episode))
        # Define an instance of the Food Pellet and initialize its locations within the game board
        food = FoodPellet()
        food.rect.x = math.floor((WINDOW_WIDTH / 4) * 3 - 20)
        food.rect.y = math.floor((WINDOW_HEIGHT / 2) - 20)

        # Reinitialize Snake's location within the game board at beginning of every epoch
        snake.trajectory = pygame.math.Vector2(BLOCK_SIZE, 0)
        snake.snake_length = 1
        snake.entire_snake = []
        snake.rect.x = math.floor((WINDOW_WIDTH / 4) - 20)
        snake.rect.y = math.floor((WINDOW_HEIGHT / 2) - 20)

        # Derive the observable state vector based on the surrounding environment information of the snake
        state_vector = Deep_Q_Snake.state_vector(food_loc=food.rect, window_width=WINDOW_WIDTH, window_height=WINDOW_HEIGHT)

        # Loop until the game ends
        steps_since_food = 0
        game_over = False
        while not game_over and steps_since_food <= MAX_STEPS_SINCE_FOOD:
            # Instantiate Reward
            reward = snake.else_reward

            # Place the Snake agent and the food instance on the game screen
            SCREEN.blit(snake.surf, snake.rect)
            SCREEN.blit(food.surf, food.rect)

            for event in pygame.event.get():
                # If the game is quit, game over and finish execution
                if event.type == pygame.QUIT:
                    game_over = True

            # Update Snake asset's trajectory based on keyboard inputs (this will later become DQL policy inputs)
            action = Deep_Q_Snake.dql_update(food_loc=food.rect, window_width=WINDOW_WIDTH, window_height=WINDOW_HEIGHT)

            # Move the snake to its new location and check whether a collision occurs with the snakes tail
            game_over = snake.move_snake()

            # Generate the state vector for the next state that the snake agent visits
            state_prime_vector = Deep_Q_Snake.state_vector(food_loc=food.rect, window_width=WINDOW_WIDTH,
                                                    window_height=WINDOW_HEIGHT)

            # If snake leaves the bounds of the board... game over
            if snake.rect.x >= WINDOW_WIDTH or snake.rect.x < 0 or snake.rect.y >= WINDOW_HEIGHT or snake.rect.y < 0:
                game_over = True

            # If a collision has been recorded or the snake has played past the number of steps since last eating food,
            # return the rewards for a collision
            if game_over or steps_since_food == MAX_STEPS_SINCE_FOOD:
                game_over = True
                reward = snake.collision_reward
                if steps_since_food == MAX_STEPS_SINCE_FOOD:
                    spin_out_count += 1

            # If the snake enters the game cell where the food is located, update the position of the food, return
            # reward for eating food and increase the length of the snake
            if snake.rect.x == food.rect.x and snake.rect.y == food.rect.y:
                food.update(WINDOW_WIDTH, WINDOW_HEIGHT, snake.entire_snake)
                reward = snake.food_reward
                snake.snake_length += 1
                steps_since_food = 0
            else:
                steps_since_food += 1

            # Record state, action, reward, and state_prime all to replay buffers to be sampled from for training
            Deep_Q_Snake.update_replay_buffer(state_vector=state_vector, action=action, reward=reward,
                                       state_prime_vector=state_prime_vector, terminal_state=game_over)

            # Fill in the background of the game in order to overwrite previous food and Snake states
            SCREEN.fill(BACKGROUND)

            # Display the updated locations of the snake and food within the game screen
            snake.display_snake(SCREEN)
            food.display_food(SCREEN)

            # Display the current score of the game
            score_message = pygame.font.SysFont(None, 35).render("Game Score: {}".format(str(snake.snake_length - 1)),
                                                                 True, (255, 255, 255))
            SCREEN.blit(score_message, [0, 0])

            # State vector for next state becomes assigned to current state vector at the end of the step
            state_vector = state_prime_vector

            # Update the display within the game window
            pygame.display.update()
            CLOCK.tick(snake.speed)

        # If game over, display screen indicating so
        game_over_message = pygame.font.SysFont(None, 50).render('GAME OVER', True, (238, 75, 43))
        SCREEN.blit(game_over_message, [WINDOW_WIDTH / 4, WINDOW_HEIGHT / 4])
        pygame.display.update()

        '''
        # To train the model, remove the block comment quotations above this line
        if snake.snake_length-1 > best_score:
            best_score = snake.snake_length-1
            snake.save_model()

        snake.train(epochs=1, verbose=1)    # '''

    # Exit the game and finish execution
    pygame.quit()
    print("Best Achieved Score: {}".format(best_score))
    print("Agent Spun Out {} Times".format(spin_out_count))
    sys.exit()
