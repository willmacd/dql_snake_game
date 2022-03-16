""" TODO """
import sys
import math
import pygame

from snake import Snake
from food import FoodPellet

BACKGROUND = (0, 0, 0)
BORDER = (200, 200, 200)
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 400
BLOCK_SIZE = 20

global SCREEN, CLOCK


def init_grid():
    """
    Initialize the game grid for the snake game

    :return:
    """

    for x in range(0, WINDOW_WIDTH, BLOCK_SIZE):
        for y in range(0, WINDOW_HEIGHT, BLOCK_SIZE):
            rect = pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(SCREEN, BORDER, rect, 1)


if __name__ == '__main__':
    pygame.init()
    SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    CLOCK = pygame.time.Clock()
    SCREEN.fill(BACKGROUND)

    snake = Snake()
    snake.rect.x = math.floor((WINDOW_WIDTH / 4) - 20)
    snake.rect.y = math.floor((WINDOW_HEIGHT / 2) - 20)

    food = FoodPellet()
    food.rect.x = math.floor((WINDOW_WIDTH / 4) * 3 - 20)
    food.rect.y = math.floor((WINDOW_HEIGHT / 2) - 20)

    while True:
        init_grid()
        SCREEN.blit(snake.surf, snake.rect)
        SCREEN.blit(food.surf, food.rect)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.update()