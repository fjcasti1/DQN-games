
import numpy as np
from games.snake import SnakeGame
import random
from collections import deque
import torch
from models.model import Linear_Qnet, QTrainer
from utils.helper import plot
from typing import Deque, List, Tuple


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 1e-3
INPUT_SIZE = 11
HIDDEN_SIZE = 256
OUTPUT_SIZE = 3


class Agent:
    "Class describing the AI agent"

    def __init__(self) -> None:
        self.n_games = 0
        self.epsilon = 0  # Control the randomness
        self.gama = 0.9  # Discount rate
        self.memory: Deque[Tuple[np.array, List[int], int, np.array, bool]] = deque(
            maxlen=MAX_MEMORY
        )
        self.model = Linear_Qnet(
            input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE
        )
        self.trainer = QTrainer(model=self.model, lr=LR, gama=self.gama)

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
        self.epsilon = max(60 - self.n_games, 0)
        move = [0, 0, 0]
        # Random move
        if random.randint(0, 200) < self.epsilon:
            idx = random.randint(0, 2)
            move[idx] = 1
        else:
            prediction = self.model(torch.tensor(state, dtype=torch.float))
            idx = torch.argmax(prediction).item()
            move[idx] = 1
        return move


def train() -> None:
    """
    Perform training
    """
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    agent = Agent()
    game = SnakeGame(game_name="Snake")

    while True:
        # get current state
        state = game.get_state()

        # get move
        action = agent.get_action(state)

        # perform the move and get new state
        game_over, reward, score = game.play_step(action)
        state_new = game.get_state()

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
                agent.model.save()

            print(f"Game: {agent.n_games} -- Score: {score} -- Record: {record}")

            plot_scores.append(score)
            total_score = score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()
