import matplotlib.pyplot as plt
import numpy as np

from data import *
from pegasos import *


def loss_convergence_depending_on_batch_size(training_data, desired_loss):
    """
    Compares the numbers of iterations required to achieve the same loss, 
    for different batch sizes.
    """
    batch_sizes = np.arange(2.5, 100, 2.5)
    iters_required = []
    for size in batch_sizes:
        params = PegasosParameters(lmbda=1, batch_size=int(
            len(training_data.data) * size / 100), iterations=10000000000, loss_threshold=desired_loss, verbose=False)
        model = PegasosSolver(params)
        iters_required.append(len(model.train(training_data)))
        print(f'{size}% done...')
    batch_sizes = [f'{size}%' for size in batch_sizes]

    _plot_iterations_required(desired_loss, batch_sizes, iters_required)


def _plot_iterations_required(desired_loss, k, iters):
    plt.figure()
    plt.title(f'Iterations required to obtain a loss of {desired_loss}')
    plt.xlabel('Batch Size')
    plt.ylabel('Iterations')
    plt.xticks(rotation=65)
    plt.bar(k, iters)
    plt.show()


def compare_loss_convergence(desired_loss: float, training_data: LabeledData, mini_batch_size: int):
    """
    Compares the rates of convergence for the mini-batch and normal Pegasos SVM solvers,
    by plotting their loss functions.
    """
    mini_batch_params = PegasosParameters(lmbda=1, batch_size=mini_batch_size,
                                          iterations=100000000, loss_threshold=desired_loss, verbose=False)
    normal_params = PegasosParameters(
        lmbda=1, iterations=100000000, loss_threshold=desired_loss, verbose=False)

    mini_batch_model = PegasosSolver(mini_batch_params)
    normal_model = PegasosSolver(normal_params)

    mini_batch_loss = mini_batch_model.train(training_data)
    normal_loss = normal_model.train(training_data)

    _plot_losses_on_same_figure(
        mini_batch_loss, normal_loss, mini_batch_size, desired_loss, len(training_data.data))


def _plot_losses_on_same_figure(mini_batch_loss, normal_loss, mini_batch_size, desired_loss, data_len):
    mini_batch_size_percentage = 100 * \
        mini_batch_size / data_len

    plt.figure()
    plt.title(f'Loss convergence - desired loss = {desired_loss}')
    plt.plot(mini_batch_loss, 'g+')
    plt.plot(normal_loss, 'r+')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend(
        [f'Mini-Batch Pegasos(k={mini_batch_size}, i.e. {mini_batch_size_percentage:.1f}%)', 'Normal Pegasos'])
    plt.savefig(f'convergence_comparison_k={mini_batch_size}')
    plt.show()


def compare_solution_accuracy(training_data, iterations, batch_sizes):
    normal_pegasos_model = PegasosSolver(PegasosParameters(
        lmbda=1, iterations=iterations, verbose=False))
    losses = [normal_pegasos_model.train(training_data)[-1]]
    classes = ['Normal Pegasos']
    for size in batch_sizes:
        model = PegasosSolver(PegasosParameters(
            lmbda=1, iterations=iterations, batch_size=size, verbose=False))
        percentage = size * 100 / len(training_data.data)
        losses.append(model.train(training_data)[-1])
        classes.append(f'Mini Batch Pegasos\nBatch Size={size},\ni.e. {percentage:.1f}%')
    _plot_solution_accuracy(classes, losses, iterations)


def _plot_solution_accuracy(classes, losses, iterations):
    plt.figure()
    plt.title(f'Loss after {iterations} iterations')
    plt.xlabel('Loss')
    plt.ylabel('Solver')
    plt.barh(classes, losses)
    plt.show()
