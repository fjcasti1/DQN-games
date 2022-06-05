from random import randint
from typing import Optional

import numpy as np
import pygame

from games.base import BaseGame
from utils.types import Direction, Point

# RGB colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20


class SnakeGame(BaseGame):
    """
    Class defining the snake game environment
    """

    def __init__(self, game_name, width: int = 800, height: int = 600):
        super().__init__(game_name, width, height)
        self.reset()

    def reset(self):
        super().reset()
        # Init Game State
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y),
        ]
        self._place_food()

    def perform_action(self, action):
        # Move, updates head future position
        self._move(action)
        self.snake.insert(0, self.head)

        # Check if Game Over
        reward = 0
        if self._is_collision() or self.iteration > 100 * len(self.snake):
            self.game_over = True
            reward = -10
        # Place new food
        elif self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        return reward

    def get_state(self):
        """
        Get game state as input for the model

        Args:
            game (SnakeGame): snake game being played

        Returns:
            np.array: 11-dimensional array containing the following information:
                * Current direction of movement
                * Food positions
                * Danger positions
        """
        head = self.head
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # Movement directions
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location
            self.food.x < self.head.x,  # Food to the left
            self.food.x > self.head.x,  # Food to the right
            self.food.y < self.head.y,  # Food to the north
            self.food.y > self.head.y,  # Food to the south
            # Danger location
            # - Danger straigth
            (dir_l and self._is_collision(point_l))
            or (dir_r and self._is_collision(point_r))
            or (dir_u and self._is_collision(point_u))
            or (dir_d and self._is_collision(point_d)),
            # - Danger left
            (dir_l and self._is_collision(point_d))
            or (dir_r and self._is_collision(point_u))
            or (dir_u and self._is_collision(point_l))
            or (dir_d and self._is_collision(point_r)),
            # - Danger right
            (dir_l and self._is_collision(point_u))
            or (dir_r and self._is_collision(point_d))
            or (dir_u and self._is_collision(point_r))
            or (dir_d and self._is_collision(point_l)),
        ]
        return np.array(state, dtype=int)

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

    def _is_collision(self, pt: Optional[Point] = None) -> bool:
        """
        Checks wether or not the new position of the head of the snake results in
        a collision.

        Returns:
            bool: True if there is a collision, False otherwise.
        """
        if pt is None:
            pt = self.head
        # Hits boundary
        if pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0:
            return True
        # Hits its own body
        elif pt in self.snake[1:]:
            return True
        else:
            return False

    def _move(self, action: np.array) -> None:
        """
        Updates the snake's head position according to the direction given.

        Args:
            direction (Direction): direction of movement
        """
        # [straght, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
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

        text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
