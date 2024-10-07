"""Pytorch."""

import nltk
import numpy as np
from numpy.typing import NDArray
import torch
from typing import List, Optional
from torch import nn
import matplotlib.pyplot as plt


FloatArray = NDArray[np.float64]


def onehot(vocabulary: List[Optional[str]], token: Optional[str]) -> FloatArray:
    """Generate the one-hot encoding for the provided token in the provided vocabulary."""
    embedding = np.zeros((len(vocabulary), 1))
    try:
        idx = vocabulary.index(token)
    except ValueError:
        idx = len(vocabulary) - 1
    embedding[idx, 0] = 1
    return embedding


def loss_fn(logp: float) -> float:
    """Compute loss to maximize probability."""
    return -logp


class Unigram(nn.Module):
    def __init__(self, V: int):
        super().__init__()

        # construct uniform initial s
        s0 = np.ones((V, 1))
        self.s = nn.Parameter(torch.tensor(s0.astype(float)))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # convert s to proper distribution p
        logp = torch.nn.LogSoftmax(0)(self.s)

        # compute log probability of input
        return torch.sum(input, 1, keepdim=True).T @ logp


def gradient_descent_example():
    """Demonstrate gradient descent."""
    # generate vocabulary
    vocabulary = [chr(i + ord("a")) for i in range(26)] + [" ", None]

    # generate training document
    text = nltk.corpus.gutenberg.raw("austen-sense.txt").lower()

    # tokenize - split the document into a list of little strings
    tokens = [char for char in text]

    # generate one-hot encodings - a V-by-T array
    encodings = np.hstack([onehot(vocabulary, token) for token in tokens])

    # convert training data to PyTorch tensor
    x = torch.tensor(encodings.astype(float))

    # define model
    model = Unigram(len(vocabulary))

    # set number of iterations and learning rate
    num_iterations =  700
    learning_rate =  0.01

    # train model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_list = []
    token_count = np.sum(encodings, axis=1)
    total_tokens = np.sum(token_count)
    optimal_probability = token_count / total_tokens
    minimum_loss = -np.sum(token_count * np.log(optimal_probability))

    for _ in range(num_iterations):
        logp_pred = model(x)
        loss = loss_fn(logp_pred)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_list.append(loss.item())
    
    # Computing learned probabilities
    with torch.no_grad():
        learned_log_probs = torch.nn.LogSoftmax(0)(model.s).numpy().flatten()
        learned_probs = np.exp(learned_log_probs)

    # Plot the final token probabilities
    plt.figure(figsize=(12, 6))
    indices = np.arange(len(vocabulary))
    width = 0.35
    plt.bar(indices - width / 2, optimal_probability, width, label='Optimal Probabilities')
    plt.bar(indices + width / 2, learned_probs, width, label='Learned Probabilities')
    plt.xticks(indices, vocabulary)
    plt.xlabel('Tokens')
    plt.ylabel('Probabilities')
    plt.title('Comparison of Optimal and Learned Token Probabilities')
    plt.legend()
    plt.show()

    # Plot the loss over iterations
    plt.figure(figsize=(10, 5))
    plt.plot(loss_list, label='Training Loss')
    plt.axhline(y=minimum_loss, color='r', linestyle='--', label='Minimum Possible Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Over Iterations')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    gradient_descent_example()
