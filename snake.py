""" TODO """
import pygame

import numpy as np


class Snake(pygame.sprite.Sprite):
    def __init__(self, block_size: int = 20):
        """
        Constructor for Snake sprite class; defining all functionality of the Snake asset in the snake game
        """
        super(Snake, self).__init__()
        # Initialize game surface for agent, and assign agent a color
        self.surf = pygame.Surface((block_size, block_size))
        self.surf.fill((0, 0, 139))

        # Define agent body within game surface
        self.rect = self.surf.get_rect()

        # Define trajectory of snake agents head
        self.__velocity = pygame.math.Vector2(block_size, 0)

    def update(self):
        """
        Update the trajectory of the snake at a given point in time based on the determined action from the DQL network

        :return:
        """
        # Get action from DQL at given point in time
        action = None

        # If policy indicates move right
        if np.argmax(action) == 1:
            pass
        # If policy indicates move left
        elif np.argmax(action) == 2:
            pass
        # Otherwise, continue straight
        else:
            pass
