from abc import ABC, abstractclassmethod
from pathlib import Path
from typing import Tuple

import numpy as np
import pygame

pygame.init()

SPEED = 40


class BaseGame(ABC):
    def __init__(self, game_name: str, width: int = 800, height: int = 600):
        """
        Basic common settings for different games

        Args:
            game_name (str): Title for the game window
            width (int, optional): Width of the game window. Defaults to 800.
            height (int, optional): Height of the game window. Defaults to 600.
        """
        self.w = width
        self.h = height
        self.font = pygame.font.Font(Path("./games/assets/arial.ttf"), 25)

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

    @abstractclassmethod
    def perform_action(self, action):
        pass

    @abstractclassmethod
    def get_state(self):
        pass

    @abstractclassmethod
    def _update_ui(self):
        pass
