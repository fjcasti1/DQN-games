from collections import namedtuple
from enum import Enum
from random import randint
from typing import Optional, Tuple

import numpy as np
import pygame

pygame.init()

SPEED = 40


class BaseGame:
    def __init__(self, game_name, width: int = 800, height: int = 600):
        self.w = width
        self.h = height

        # Init Display
        self.display = pygame.display.set_mode(size=(self.w, self.h))
        pygame.display.set_caption(game_name)
        self.clock = pygame.time.Clock()

    def reset(self):
        self.game_over = False
        self.score = 0
        self.iteration = 0

    def play_step(self, action: np.array) -> Tuple[bool, int, int]:
        """
        Evaluates the action of the player in the game and it's possible outcomes.

        Returns:
            Tuple[bool, int]: game_over, score
        """
        self.iteration += 1

        # Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Perform action
        reward = self.perform_action(action)

        # Update UI and Clock
        self._update_ui()
        self.clock.tick(SPEED)

        return self.game_over, reward, self.score

    def perform_action(action):
        pass

    def get_state(self):
        pass

    def _update_ui(self):
        pass