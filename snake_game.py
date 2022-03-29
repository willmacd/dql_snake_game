""" Main file for Deep Q-Learning Snake Game Implementation """
import sys
import math
import time
import pygame

from snake import Snake
from food import FoodPellet

BACKGROUND = (0, 0, 0)
BORDER = (200, 200, 200)
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400
BLOCK_SIZE = 20

global SCREEN, CLOCK


if __name__ == '__main__':
    # Initialize an instance of pygame for development
    pygame.init()

    # Instantiate a window containing the game board and define a game clock
    SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    SCREEN.fill(BACKGROUND)
    CLOCK = pygame.time.Clock()

    # Update display and set a Window name for the game
    pygame.display.update()
    pygame.display.set_caption("Deep Q-Learning Snake Game")

    # Define an instance of the Food Pellet and initialize its locations within the game board
    food = FoodPellet()
    food.rect.x = math.floor((WINDOW_WIDTH / 4) * 3 - 20)
    food.rect.y = math.floor((WINDOW_HEIGHT / 2) - 20)

    # Define an instance of the Snake agent and initialize its location within the game board
    snake = Snake()
    snake.rect.x = math.floor((WINDOW_WIDTH / 4) - 20)
    snake.rect.y = math.floor((WINDOW_HEIGHT / 2) - 20)

    # Loop until the game ends
    game_over = False
    while not game_over:
        # Place the Snake agent and the food instance on the game screen
        SCREEN.blit(snake.surf, snake.rect)
        SCREEN.blit(food.surf, food.rect)

        for event in pygame.event.get():
            # If the game is quit, game over and finish execution
            if event.type == pygame.QUIT:
                game_over = True

            # If an action is received update the snakes position within the environment
            if event.type == pygame.KEYDOWN:
                # Update Snake asset's trajectory based on keyboard inputs (this will later become DQL policy inputs)
                snake.update(event=event)

        # Derive the observable state vector based on the surronding environment information of the snake
        state_vector = snake.state_vector(food_loc=food.rect, window_width=WINDOW_WIDTH, window_height=WINDOW_HEIGHT)

        # Move the snake to its new location and check whether a collision occurs with the snakes tail
        game_over = snake.move_snake()
        SCREEN.fill(BACKGROUND)

        # If snake leaves the bounds of the board... game over
        if snake.rect.x >= WINDOW_WIDTH or snake.rect.x < 0 or snake.rect.y >= WINDOW_HEIGHT or snake.rect.y < 0:
            game_over = True

        # Display the updated locations of the snake and food within the game screen
        snake.display_snake(SCREEN)
        food.display_food(SCREEN)


        # Display the current score of the game
        score_message = pygame.font.SysFont(None, 35).render("Game Score: {}".format(str(snake.snake_length - 1)),
                                                             True, (255, 255, 255))
        SCREEN.blit(score_message, [0, 0])

        # Update the display within the game window
        pygame.display.update()

        # If the snake enters the game cell where the food is located, update the position of the food
        if snake.rect.x == food.rect.x and snake.rect.y == food.rect.y:
            food.update(WINDOW_WIDTH, WINDOW_HEIGHT, snake.entire_snake)
            snake.snake_length += 1

        CLOCK.tick(snake.speed)

    # If game over, display screen indicating so
    game_over_message = pygame.font.SysFont(None, 50).render('GAME OVER', True, (238, 75, 43))
    SCREEN.blit(game_over_message, [WINDOW_WIDTH / 4, WINDOW_HEIGHT / 4])
    pygame.display.update()
    time.sleep(2)

    # Exit the game and finish execution
    pygame.quit()
    sys.exit()
