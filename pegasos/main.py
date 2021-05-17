import matplotlib.pyplot as plt

from data import *
from dataclasses import dataclass
from mini_batch_pegasos import *

def plot_loss(loss, batch_size):
    plt.figure()
    plt.title(f'Loss evolution for k = {batch_size}')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.plot(loss)
    plt.show()

if __name__ == '__main__':
    training_data = read_and_relabel_train_data('MNIST-13.csv', 1)
    params = MiniBatchParameters(lmbda=1, batch_size = 30, iterations=1000)
    model = MiniBatchPegasos(training_data.labels, params)
    loss = model.train(training_data)
    plot_loss(loss, 30)