import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from numpy import linalg as la
from scipy import optimize

# min_x (x1 - 2)^4 + (x1 - 2x2)^2
# s.l. x in Q = {||x|| <= 2}


def function(x: np.array) -> float:
    return (x[0][0] - 2)**4 + (x[0][0] - 2*x[1][0])**2


def gradient(x: np.array) -> np.array:
    return np.array([[4*(x[0][0] - 2)**3 + 2*(x[0][0] - 2*x[1][0])], [-4*(x[0][0] - 2*x[1][0])]])


def projection(x: np.array) -> np.array:
    return min(1, 2/(sqrt(x[0]**2 + x[1]**2))) * x


def stop(xk: np.array, epsilon: float) -> bool:
    grad = gradient(xk)
    return sqrt(grad[0][0]**2 + grad[1][0]**2) <= epsilon


epsilon = 10**-1
cons = ({'type': 'ineq', 'fun': lambda x: -1 * sqrt(x[0]**2 + x[1]**2) + 2})
f_star = optimize.minimize(lambda x: (
    x[0] - 2)**4 + (x[0] - 2*x[1])**2, (1, 1), method='SLSQP', constraints=cons)
target = function(np.array([[f_star.x[0]], [f_star.x[1]]]))


def const_projected_gradient_descent(alpha: float = 0.1):
    new_x, old_x, steps = projection(
        np.array([[1], [1]])), projection(np.array([[1], [1]])), 0
    gradient_norm_evolution = []
    function_evolution = []
    while not stop(new_x, epsilon):
        new_x = projection(old_x - alpha * gradient(old_x))
        old_x = new_x
        grad = gradient(new_x)
        gradient_norm_evolution.append(sqrt(grad[0][0]**2 + grad[1][0]**2))
        function_evolution.append(abs(function(new_x) - target))
        steps += 1
    return old_x, steps, gradient_norm_evolution, function_evolution


def backtracking_projected_gradient_descent():
    new_x, old_x, steps = projection(
        np.array([[1], [1]])), projection(np.array([[1], [1]])), 0
    gradient_norm_evolution = []
    function_evolution = []
    c, ro = 0.5, 0.1
    while not stop(new_x, epsilon):
        alpha = 0.5
        new_x = projection(old_x - alpha * gradient(old_x))
        while function(new_x) > function(old_x) - (c/alpha) * ((new_x[0][0] - old_x[0][0])**2 - (new_x[1][0] - old_x[1][0])**2):
            alpha *= ro
            new_x = projection(old_x - alpha * gradient(old_x))
        new_x = projection(old_x - alpha * gradient(old_x))
        old_x = new_x
        grad = gradient(new_x)
        gradient_norm_evolution.append(sqrt(grad[0][0]**2 + grad[1][0]**2))
        function_evolution.append(abs(function(new_x) - target))
        steps += 1
    return old_x, steps, gradient_norm_evolution, function_evolution


if __name__ == '__main__':
    _, const_steps, const_gradient_norm_evolution, const_function_evolution = const_projected_gradient_descent()
    _, back_steps, back_gradient_norm_evolution, back_function_evolution = backtracking_projected_gradient_descent()

    plt.plot([i for i in range(1, const_steps + 1)],
             const_gradient_norm_evolution)
    plt.plot([i for i in range(1, back_steps + 1)],
             back_gradient_norm_evolution)
    plt.ylabel("||gradient(xk)||")
    plt.xlabel("k")
    plt.legend(["Constant Step(alpha = 0.1)", "Backtracking Step"])
    plt.show()
    plt.savefig("gradient_evolution")
    
    plt.plot([i for i in range(1, const_steps + 1)],
             const_function_evolution)
    plt.plot([i for i in range(1, back_steps + 1)],
             back_function_evolution)
    plt.ylabel("||f(xk) - f*||")
    plt.xlabel("k")
    plt.legend(["Constant Step(alpha = 0.1)", "Backtracking Step"])
    plt.show()
    plt.savefig("difference_to_target_evolution")