""" Class file containing necessary functionality for the Food Pellet asset within the Snake game """
import random
import pygame


class FoodPellet(pygame.sprite.Sprite):
    def __init__(self, block_size: int = 20):
        """
        Constructor for FoodPellet sprite class, defining all functionality of the FoodPellet asset in the snake game

        :param block_size: Size of single square within the game board
        """
        super(FoodPellet, self).__init__()
        # Initialize game surface for agent, and assign asset a color
        self.surf = pygame.Surface((block_size, block_size))
        self.surf.fill((238, 75, 43))

        # Define attribute to define the size of game board cells
        self.block_size = block_size

        # Define asset within game surface
        self.rect = self.surf.get_rect()

    def update(self, window_width, window_height, snake_body):
        """
        Update position of the Food Pellet to a random location not currently occupied by any part of the snake

        :param window_width: Width of the game window in which the food can be spawned
        :param window_height: Height of the game window in which the food can be spawned
        :param snake_body: Coordinates for all cells belonging to the snakes body
        :return:
        """
        self.rect.x = round(random.randrange(0, window_width - self.block_size) / self.block_size) * self.block_size
        self.rect.y = round(random.randrange(0, window_height - self.block_size) / self.block_size) * self.block_size
        while [self.rect.x, self.rect.y] in snake_body:
            self.rect.x = round(
                random.randrange(0, window_width - self.block_size) / self.block_size) * self.block_size
            self.rect.y = round(
                random.randrange(0, window_height - self.block_size) / self.block_size) * self.block_size

    def display_food(self, game_screen):
        """
        Function for displaying the Food Pellet within the game screen

        :param game_screen: Reference to window in which the game is being hosted
        :return:
        """
        pygame.draw.rect(game_screen, (238, 75, 43), [self.rect.x, self.rect.y, self.block_size, self.block_size])
