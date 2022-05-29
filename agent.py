import random
from collections import deque
from typing import Deque, List, Tuple

import numpy as np
import torch

from game import BLOCK_SIZE, Direction, Point, SnakeGameAI

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 1e-3


class Agent:
    """
    Class describing the AI agent
    """

    def __init__(self) -> None:
        self.n_games = 0
        self.epsilon = 0  # Control the randomness
        self.gamma = 0  # discount rate
        self.memory: Deque[Tuple[np.array, List[int], int, np.array, bool]] = deque(
            maxlen=MAX_MEMORY
        )
        self.model = None  # TODO
        self.trainer = None  # TODO

    def get_state(self, game: SnakeGameAI) -> np.array:
        """
        Get game state as input for the model

        Args:
            game (SnakeGameAI): snake game being played

        Returns:
            np.array: 11-dimensional array containing the following information:
                * Current direction of movement
                * Food positions
                * Danger positions
        """
        head = game.head
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Movement directions
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location
            game.food.x < game.head.x,  # Food to the left
            game.food.x > game.head.x,  # Food to the right
            game.food.y < game.head.y,  # Food to the north
            game.food.y > game.head.y,  # Food to the south
            # Danger location
            # - Danger straigth
            (dir_l and game.is_collision(point_l))
            or (dir_r and game.is_collision(point_r))
            or (dir_u and game.is_collision(point_u))
            or (dir_d and game.is_collision(point_d)),
            # - Danger left
            (dir_l and game.is_collision(point_d))
            or (dir_r and game.is_collision(point_u))
            or (dir_u and game.is_collision(point_l))
            or (dir_d and game.is_collision(point_r)),
            # - Danger right
            (dir_l and game.is_collision(point_u))
            or (dir_r and game.is_collision(point_d))
            or (dir_u and game.is_collision(point_r))
            or (dir_d and game.is_collision(point_l)),
        ]
        return np.array(state, dtype=int)

    def remember(
        self,
        state: np.array,
        action: List[int],
        reward: int,
        next_state: np.array,
        game_over: bool,
    ) -> None:
        """
        Store training variables in deque memory

        Args:
            state (np.array): 11-dimensional array containing state information
            action (List[int]): List describing the next movement
            reward (int): Reward value
            next_state (np.array): 11-dimensional array containing next state
                information
            game_over (bool): Wether or not the AI lost the game
        """
        self.memory.append(
            (state, action, reward, next_state, game_over)
        )  # popleft if MAX_MEMORY is reached

    def train_long_memory(self) -> None:
        """
        Perform a training step with all collected values (long term memory)
        """
        if len(self.memory) > BATCH_SIZE:
            batch = deque(random.sample(self.memory, BATCH_SIZE))  # List of tuples
        else:
            batch = self.memory

        states, actions, rewards, next_states, game_overs = zip(*batch)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

    def train_short_memory(
        self,
        state: np.array,
        action: List[int],
        reward: int,
        next_state: np.array,
        game_over: bool,
    ) -> None:
        """
        Perform a training step with individual values (short term memory)

        Args:
            state (np.array): 11-dimensional array containing state information
            action (List[int]): List describing the next movement
            reward (int): Reward value
            next_state (np.array): 11-dimensional array containing next state
                information
            game_over (bool): Wether or not the AI lost the game
        """
        self.trainer.train_step(state, action, reward, next_state, game_over)

    def get_action(self, state: np.array) -> List[int]:
        """
        Get the following movement either via prediction or randomly generated

        Args:
            state (np.array): 11-dimensional array containing state information

        Returns:
            action (List[int]): List describing the next movement
        """
        # random moves: traeeoff exploration / exploitation
        self.epsilon = max(80 - self.n_games, 0)
        move = [0, 0, 0]
        # Random move
        if random.randint(0, 200) < self.epsilon:
            idx = random.randint(0, 2)
            move[idx] = 1
        else:
            prediction = self.model.predict(torch.tensor(state, dtype=torch.float))
            idx = torch.argmax(prediction).item()
            move[idx] = 1
        return move


def train() -> None:
    """
    Perform training
    """
    # plot_scores = []
    # plot_mean_scores = []
    # total_score = 0
    record = 0

    agent = Agent()
    game = SnakeGameAI()

    while True:
        # get current state
        state = agent.get_state(game)

        # get move
        action = agent.get_action(state)

        # perform the move and get new state
        game_over, reward, score = game.play_step(action)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state, action, reward, state_new, game_over)

        # remember
        agent.remember(state, action, reward, state_new, game_over)

        if game_over:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                # TODO(Kiko): agent.model.save()

            print(f"Game: {agent.n_games} -- Score: {score} -- Record: {record}")


if __name__ == "__main__":
    train()
