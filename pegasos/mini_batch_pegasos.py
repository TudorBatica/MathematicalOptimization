import math
import numpy as np

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

    def train(self, training_data):
        weights = np.zeros([1, training_data.data.shape[1]])
        losses = []

        for t in range(1, self.params.iterations + 1):
            weights = self._compute_next_weights(weights, t, training_data)
            current_loss = self._compute_loss(training_data.data, training_data.labels, weights)
            if self.params.loss_threshold is not None and current_loss < self.params.loss_threshold:
                break
            losses.append(current_loss)
            if not t % 25: 
                print(f'Iteration {t}, loss = {current_loss}')

        return losses

    def _compute_next_weights(self, weights, iteration, training_data):
        A = self._select_batch(training_data.data, training_data.labels)
        A_plus = self._extract_examples_with_non_zero_losses(A, weights)
        learning_rate = 1.0 / (self.params.lmbda * iteration)
        weights = self._stochastic_subgradient_descent(weights, A_plus, learning_rate)
        weights = self._project_weights(weights)
        
        return weights

    def _select_batch(self, data, target):
        # this function samples k random points from both classes
        index = np.arange(0, data.shape[0]).reshape([data.shape[0], 1])
        # index of points of first class
        index_0 = index[target == self.classes[0]]
        # index of points of second class
        index_1 = index[target == self.classes[1]]
        d0 = data[index_0, :]  # data of points of first class
        t0 = target[index_0]
        d1 = data[index_1, :]  # data of points of second class
        t1 = target[index_1]
        # choose k points from each class
        index_choice_0 = np.random.choice(d0.shape[0], math.ceil(
            (self.params.batch_size / data.shape[0]) * d0.shape[0]), replace=False)
        index_choice_1 = np.random.choice(d1.shape[0], math.ceil(
            (self.params.batch_size / data.shape[0]) * d1.shape[0]), replace=False)
        # create data and target vectors to be used in the SVM
        A_data = np.empty([0, data.shape[1]])
        A_target = np.empty([0, target.shape[1]])

        A_data = np.append(A_data, d0[index_choice_0, :], axis=0)
        A_data = np.append(A_data, d1[index_choice_1, :], axis=0)
        A_target = np.append(A_target, t0[index_choice_0, :], axis=0)
        A_target = np.append(A_target, t1[index_choice_1, :], axis=0)
        
        return LabeledData(data=A_data, labels=A_target)

    def _extract_examples_with_non_zero_losses(self, set: LabeledData, weights) -> LabeledData:
        indexes = np.arange(0, set.data.shape[0])
        indexes = indexes[((set.labels * np.dot(set.data, weights.T))
                          < 1).reshape(indexes.shape)]
        
        return LabeledData(data=set.data[indexes, :], labels=set.labels[indexes, :])

    def _stochastic_subgradient_descent(self, weights, A_plus, learning_rate):
        return (1. - learning_rate * self.params.lmbda) * weights + (learning_rate / self.params.batch_size) * \
                np.sum(np.multiply(A_plus.labels, A_plus.data), axis=0)

    def _project_weights(self, weights):
        return np.minimum(np.float64(1.0), 1.0 /
                           (np.sqrt(self.params.lmbda) * np.linalg.norm(weights))) * weights

    def _compute_loss(self, data, target, w):
        # calculates the cost function of the model based on the data and weights
        loss = 0
        temp = target * np.dot(data, w.T)
        for i in range(data.shape[0]):
            loss += np.maximum(0, 1 - temp[i])  # compute hinge loss
        loss = loss / data.shape[0]
        # add regularization term
        return (loss + 0.5 * self.params.lmbda * np.linalg.norm(w) ** 2)