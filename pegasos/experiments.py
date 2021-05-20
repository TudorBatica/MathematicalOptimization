import matplotlib.pyplot as plt

from data import *
from pegasos import *

def compare_loss_convergence(desired_loss: float, training_data: LabeledData, mini_batch_size: int):
    """
    Compares the rates of convergence for the mini-batch and normal Pegasos SVM solvers,
    by plotting their loss functions.
    """
    mini_batch_params = PegasosParameters(lmbda=1, batch_size=mini_batch_size,
                                   iterations=100000000, loss_threshold=desired_loss, verbose=False)
    normal_params = PegasosParameters(lmbda=1, iterations=100000000, loss_threshold=desired_loss, verbose=False)
    
    mini_batch_model = PegasosSolver(mini_batch_params)
    normal_model = PegasosSolver(normal_params)

    mini_batch_loss = mini_batch_model.train(training_data)
    normal_loss = normal_model.train(training_data)

    mini_batch_size_percentage =  100 * mini_batch_size / len(training_data.data)

    plt.figure()
    plt.title(f'Loss convergence - desired loss = {desired_loss}')
    plt.plot(mini_batch_loss, 'g+')
    plt.plot(normal_loss, 'r+')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend([f'Mini-Batch Pegasos(k={mini_batch_size}, i.e. {mini_batch_size_percentage:.1f}%)', 'Normal Pegasos'])
    plt.savefig(f'convergence_comparison_k={mini_batch_size}')
    plt.show()

