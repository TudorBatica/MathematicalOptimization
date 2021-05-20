import experiments
import matplotlib.pyplot as plt

from data import read_and_relabel_train_data

def plot_loss(loss, batch_size):
    plt.figure()
    plt.title(f'Loss convergence; batch size = {batch_size}, lambda = 1, 1000 iterations')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.plot(loss)
    plt.show()

def iterations_required_plot(k, iters):
    plt.figure()
    plt.title(f'Iterations required to obtain a loss of 0.25 w.r.t to the batch size')
    plt.xlabel('Batch Size')
    plt.ylabel('Iterations')
    plt.bar(k, iters)
    plt.show()

if __name__ == '__main__':
    training_data = read_and_relabel_train_data('MNIST-13.csv', 1)
    
    print('Running loss convergence comparison')
    experiments.compare_loss_convergence(0.25, training_data, 50)

    
    """
    training_data = read_and_relabel_train_data('MNIST-13.csv', 1)
    params = PegasosParameters(lmbda=1, iterations=1000)
    model = PegasosSolver(params)
    loss = model.train(training_data)
    plot_loss(loss, 'Full')

    params = PegasosParameters(lmbda=1, batch_size = 30, iterations=1000)
    model = PegasosSolver(params)
    loss = model.train(training_data)
    plot_loss(loss, 30)
    """
    

    """
    iters = []
    batch_sizes = []
    for k in range(1, 300, 10):
        params = PegasosParameters(lmbda=1, batch_size = k, iterations=10000000000, loss_threshold = 0.5)
        model = PegasosSolver(training_data.labels, params)
        iters.append(len(model.train(training_data)))
        batch_sizes.append(k)

    iterations_required_plot([str(b) for b in batch_sizes], iters)
    """