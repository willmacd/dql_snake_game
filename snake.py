""" Class file containing necessary functionality for the Snake asset within the Snake game """
import pygame


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