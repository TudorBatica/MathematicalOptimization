import math
from typing import List
import numpy as np
import random

from dataclasses import dataclass
from data import LabeledData

@dataclass
class MiniBatchParameters:
    lmbda: float
    batch_size: int
    iterations: int
    loss_threshold: float = None

class MiniBatchPegasos:
    def __init__(self, target, params: MiniBatchParameters):
        self.params = params
        self.classes = np.unique(target)

    def train(self, training_data: LabeledData) -> List[float]:
        weights = np.zeros([1, training_data.data.shape[1]])
        print(weights)
        print(weights.T)
        losses = []
        for t in range(1, self.params.iterations + 1):
            weights = self._compute_next_weights(weights, t, training_data)
            current_loss = self._compute_loss(training_data, weights)
            if self.params.loss_threshold is not None and current_loss < self.params.loss_threshold:
                break
            losses.append(current_loss)
            if not t % 25: 
                print(f'Iteration {t}, loss = {current_loss}')

        return losses

    def _compute_next_weights(self, weights, iteration, training_data):
        """
        Performs a Pegasos iteration
        """
        A = self._select_batch(training_data)
        A_plus = self._extract_examples_with_non_zero_losses(A, weights)
        learning_rate = 1.0 / (self.params.lmbda * iteration)
        weights = self._stochastic_subgradient_descent(weights, A_plus, learning_rate)
        weights = self._project_weights(weights)
        
        return weights

    def _select_batch(self, training_data):
        """
        Randomly selects a mini batch of size `self.params.batch_size`
        """
        indexes = random.sample(range(0, len(training_data.data)), self.params.batch_size)
        return LabeledData(data=training_data.data[indexes], labels=training_data.labels[indexes])

    def _extract_examples_with_non_zero_losses(self, set: LabeledData, weights) -> LabeledData:
        """
        Computes the A plus set.
        """
        indexes = np.arange(0, set.data.shape[0])
        indexes = indexes[((set.labels * np.dot(set.data, weights.T))
                          < 1).reshape(indexes.shape)]
        
        return LabeledData(data=set.data[indexes, :], labels=set.labels[indexes, :])

    def _stochastic_subgradient_descent(self, weights, A_plus, learning_rate):
        """
        Performs a stochastic subgradient descent step.
        """
        return (1. - learning_rate * self.params.lmbda) * weights + (learning_rate / self.params.batch_size) * \
                np.sum(np.multiply(A_plus.labels, A_plus.data), axis=0)

    def _project_weights(self, weights):
        """
        Performs a projection step.
        """
        return np.minimum(np.float64(1.0), 1.0 /
                           (np.sqrt(self.params.lmbda) * np.linalg.norm(weights))) * weights

    def _compute_loss(self, training_data, weights):
        """
        Compute the loss function.
        """
        loss = 0
        temp = training_data.labels * np.dot(training_data.data, weights.T)
        for i in range(training_data.data.shape[0]):
            loss += np.maximum(0, 1 - temp[i])  

        return loss / training_data.data.shape[0] + self.params.lmbda / 2 * np.linalg.norm(weights) ** 2