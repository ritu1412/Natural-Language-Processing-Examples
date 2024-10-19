"""RNN example."""

import random
from typing import Mapping, Sequence
import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn

FloatArray = NDArray[np.float64]


class SimpleRNN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.Wx = nn.Linear(input_size, output_size, bias=False)
        self.Wh = nn.Linear(output_size, output_size, bias=False)

    def forward(self, document: Sequence[torch.Tensor]) -> torch.Tensor:
        output = torch.zeros((self.Wh.in_features, 1), requires_grad=True)
        for token_embedding in document:
            output = self.forward_cell(token_embedding, output)
        return torch.squeeze(output)

    def forward_cell(
        self, token_embedding: torch.Tensor, previous_output: torch.Tensor
    ) -> torch.Tensor:
        return self.Wx(token_embedding.T) + self.Wh(previous_output)


def generate_observation(length: int) -> tuple[list[str], float]:
    document = [random.choice(("good", "bad", "uh","nevermind")) for _ in range(length)]
    sentiment = 0.0
    for token in document:
        if token == "good":
            sentiment += 1
        elif token == "bad":
            sentiment += -1
    return document, sentiment


def onehot(vocabulary_map: Mapping[str, int], token: str) -> FloatArray:
    """Generate the one-hot encoding for the provided token in the provided vocabulary."""
    embedding = np.zeros((len(vocabulary_map), 1))
    idx = vocabulary_map.get(token, len(vocabulary_map) - 1)
    embedding[idx, 0] = 1
    return embedding


def rnn_example() -> None:
    """Demonstrate a simple RNN."""
    vocabulary = ["bad", "good", "uh","nevermind"]
    vocabulary_map = {token: idx for idx, token in enumerate(vocabulary)}

    # generate training data
    observation_count = 100
    max_length = 10
    observations = [
        generate_observation(int(np.ceil(np.random.rand() * max_length)))
        for _ in range(observation_count)
    ]
    X = [
        [
            torch.tensor(onehot(vocabulary_map, token).astype("float32"))
            for token in sentence
        ]
        for sentence, _ in observations
    ]

    y_true = [torch.tensor(label) for _, label in observations]

    # define model
    model = SimpleRNN(4, 1)
    loss_fn = torch.nn.MSELoss()

    # print initial parameters and loss
    print(
        list(model.parameters()),
        torch.sum(
            torch.tensor(tuple(loss_fn(model(x_i), y_i) for x_i, y_i in zip(X, y_true)))
        ),
    )

    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(100):  # loop over gradient descent steps
        for x_i, y_i in zip(X, y_true):  # loop over observations/"documents"
            y_pred = model(x_i)
            loss = loss_fn(y_pred, y_i)
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # print final parameters and loss
    print(
        list(model.parameters()),
        torch.sum(
            torch.tensor(tuple(loss_fn(model(x_i), y_i) for x_i, y_i in zip(X, y_true)))
        ),
    )


if __name__ == "__main__":
    rnn_example()
