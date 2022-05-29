from collections import namedtuple
from enum import Enum
from random import randint
from typing import Tuple

import pygame

pygame.init()
font = pygame.font.Font("arial.ttf", 25)


# reset
# reward
# play =(action) -> direction
# game_iteration
# is_collision


class Direction(Enum):
    """
    Enumeration class with the possible movement directions
    """

    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4


Point = namedtuple("Point", "x, y")

# RGB colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 20


class SnakeGameAI:
    """
    Class defining the snake game environment
    """

    def __init__(self, width: int = 1000, height: int = 800):
        self.w = width
        self.h = height

        # Init Display
        self.display = pygame.display.set_mode(size=(self.w, self.h))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self) -> None:
        """
        Resets the game
        """
        # Init Game State
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y),
        ]

        self.score = 0
        self._place_food()
        self.iteration = 0

    def _place_food(self) -> None:
        """
        Place food in a random position in the play area. If the placement lands
        in a position occupied by the snake, it is replaced
        """
        x = BLOCK_SIZE * randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE)
        y = BLOCK_SIZE * randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE)
        self.food = Point(x, y)
        # Try again if we placed food inside the snake
        if self.food in self.snake:
            self._place_food()

    def _is_collision(self) -> bool:
        """
        Checks wether or not the new position of the head of the snake results in
        a collision.

        Returns:
            bool: True if there is a collision, False otherwise.
        """
        # Hits boundary
        if (
            self.head.x >= self.w
            or self.head.x < 0
            or self.head.y >= self.h
            or self.head.y < 0
        ):
            return True
        # Hits its own body
        elif self.head in self.snake[1:]:
            return True
        else:
            return False

    def _move(self, direction: Direction) -> None:
        """
        Updates the snake's head position according to the direction given.

        Args:
            direction (Direction): direction of movement
        """
        x = self.head.x
        y = self.head.y

        if direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE

        self.head = Point(x, y)

    def _update_ui(self) -> None:
        """
        Updates the UI after a movement has occured.
        """
        # Draw background
        self.display.fill(BLACK)

        # Draw snake
        for part in self.snake:
            pygame.draw.rect(
                surface=self.display,
                color=BLUE1,
                rect=pygame.Rect(part.x, part.y, BLOCK_SIZE, BLOCK_SIZE),
            )
            pygame.draw.rect(
                surface=self.display,
                color=BLUE2,
                rect=pygame.Rect(
                    part.x + BLOCK_SIZE / 5,
                    part.y + BLOCK_SIZE / 5,
                    3 * BLOCK_SIZE / 5,
                    3 * BLOCK_SIZE / 5,
                ),
            )

        # Draw food
        pygame.draw.rect(
            surface=self.display,
            color=RED,
            rect=pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE),
        )

        text = font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def play_step(self) -> Tuple[bool, int]:
        """
        Evaluates the action of the player in the game and it's possible outcomes.

        Returns:
            Tuple[bool, int]: game_over, score
        """
        # Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN

        # Move, updates head future position
        self._move(self.direction)
        self.snake.insert(0, self.head)

        # Check if Game Over
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score

        # Place new food
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        # Update UI and Clock
        self._update_ui()
        self.clock.tick(SPEED)

        return game_over, self.score


if __name__ == "__main__":
    game = SnakeGameAI()

    # Game loop
    while True:
        game_over, score = game.play_step()
        if game_over:
            break

    pygame.quit()
