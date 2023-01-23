from typing import Callable, Dict, Protocol

import torch
import torch.nn as nn

from loguru import logger

Tensor = torch.Tensor


class GenericModel(Protocol):
    train: Callable
    eval: Callable
    parameters: Callable

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        pass


class Linear(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(config["input"], config["h1"]),
            nn.ReLU(),
            nn.Linear(config["h1"], config["h2"]),
            nn.Dropout(config["dropout"]),
            nn.ReLU(),
            nn.Linear(config["h2"], config["output"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=1)
        x = self.encoder(x)
        return x


class GRUmodel(nn.Module):
    def __init__(self, config: Dict) -> None:
        super(GRUmodel, self).__init__()
        self.rnn = nn.GRU(
            input_size=config["input"],
            hidden_size=config["hidden_size"],
            dropout=config["dropout"],
            batch_first=True,
            num_layers=config["num_layers"],
        )
        self.linear1 = nn.Linear(config["hidden_size"], config["output"])
        self.linear2 = nn.Linear(config["output"], config["num_classes"])
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        last_step = x[:, -1, :].squeeze() 
        yhat = self.linear1(last_step)
        yhat = self.relu(yhat)
        yhat = self.linear2(yhat)
        yhat = self.softmax(yhat)
        return yhat

class GRUClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRUClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :]) # take the last hidden state
        return out

class BaseRNN(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, horizon: int
    ) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=num_layers,
        )
        self.linear = nn.Linear(hidden_size, horizon)
        self.horizon = horizon

    def forward(self, x: Tensor) -> Tensor:
        x, _ = self.rnn(x)
        last_step = x[:, -1, :]
        yhat = self.linear(last_step)
        return yhat



class Accuracy:
    def __repr__(self) -> str:
        return "Accuracy"

    def __call__(self, y: Tensor, yhat: Tensor) -> Tensor:
        """
        yhat is expected to be a vector with d dimensions.
        The highest values in the vector corresponds with
        the correct class.
        """
        return (yhat.argmax(dim=1) == y).sum() / len(yhat)
