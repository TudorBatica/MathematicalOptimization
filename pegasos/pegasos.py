import math
from typing import List
import numpy as np
import random

from dataclasses import dataclass
from data import LabeledData

@dataclass
class PegasosParameters:
    lmbda: float
    iterations: int
    batch_size: int = None
    loss_threshold: float = None
    verbose: bool = True

class PegasosSolver:
    def __init__(self, params: PegasosParameters):
        self.params = params

    def train(self, training_data: LabeledData) -> List[float]:
        weights, losses = self._initialize(training_data), [] 
        
        for t in range(1, self.params.iterations + 1):
            weights = self._compute_next_weights(weights, t, training_data)
            current_loss = self._compute_loss(training_data, weights)
            losses.append(current_loss)
            
            if self.params.verbose and not t % 25: 
                self._print_iteration_details(current_loss, losses[-2], t)
            
            if self.params.loss_threshold is not None and current_loss < self.params.loss_threshold:
                break

        return losses

    def _initialize(self, training_data):
        """
        Returns the initial values for the weights vector and
        initializes all other params.
        """
        if not self.params.batch_size:
            self.params.batch_size = len(training_data.data)

        return np.zeros([1, training_data.data.shape[1]])

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
        if self.params.batch_size == len(training_data.data):
            return training_data
        
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
        temp = training_data.labels * np.dot(training_data.data, weights.T)
        loss = self.params.lmbda / 2 * np.linalg.norm(weights) ** 2 \
            + sum([np.maximum(0, 1 - p) for p in temp]) / len(training_data.data)

        return loss[0]
    
    def _print_iteration_details(self, current_loss, previous_loss, iteration):
        difference = max(current_loss, previous_loss) * 100 / min(current_loss, previous_loss)
        message = 'Increase' if current_loss > previous_loss else 'Decrease'

        print(f'Iteration {iteration} ---> Loss is {current_loss:.4f} ---> {difference:.2f}% {message}')