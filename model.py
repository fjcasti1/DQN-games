import os
from typing import List, Union

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim


class Linear_Qnet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name: str = "model.pth") -> None:
        model_folder_path = "./store"
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr: float, gama: float) -> None:
        self.lr = lr
        self.gama = gama
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        game_over: Union[bool, List[bool]],
    ) -> None:
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            game_over = (game_over,)

        # 1: predicted Q values with the current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(state.shape[0]):
            Q_new = reward[idx]
            if not game_over[idx]:
                Q_new = reward[idx] + self.gama * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action).item()] = Q_new

        # 2: Q_new + gama * max( next_predicted Q value ) -> only do this if not game_over
        # pred.close()
        # pred[argmac(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()
