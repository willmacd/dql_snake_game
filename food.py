""" TODO """

import pygame


class FoodPellet(pygame.sprite.Sprite):
    def __init__(self):
        """
        Constructor for FoodPellet sprite class, defining all functionality of the FoodPellet asset in the snake game
        """
        super(FoodPellet, self).__init__()
        # Initialize game surface for agent, and assign asset a color
        self.surf = pygame.Surface((20, 20))
        self.surf.fill((238, 75, 43))

        # Define asset within game surface
        self.rect = self.surf.get_rect()

    def update(self):
        """
        Update position of the Food Pellet to a random location not currently occupied by any part of the snake

        :return:
        """
        eaten = False

        # If snake collides with food pellet at given point in time, spawn new pellet randomly within the board such
        # that it is not overlapping with any part of the snake
        if eaten:
            pass
        # Otherwise, leave the food pellet at its current location
        else:
            pass
