import experiments
import matplotlib.pyplot as plt

from data import read_and_relabel_train_data

if __name__ == '__main__':
    training_data = read_and_relabel_train_data(
        'dataset/MNIST-13.csv', negativeLabelClass=1)

    #print('Running loss convergence depending on batch size')
    #experiments.loss_convergence_depending_on_batch_size(desired_loss=0.25, training_data=training_data)

    print('Running loss convergence comparison')
    experiments.compare_loss_convergence(0.25, training_data, 75)

    #print('Running solution accuracy comparison')
    #experiments.compare_solution_accuracy(training_data, 1000, [50, 150])
