import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from numpy import linalg as la
from scipy import optimize

# min_x (x1 - 2)^4 + (x1 - 2x2)^2
# s.l. x in Q = {||x|| <= 2}


class Result:
    def __init__(self, solution, steps, gradient_norm_evolution, distance_to_target_evolution):
        self.solution = solution
        self.steps = steps
        self.gradient_norm_evolution = gradient_norm_evolution
        self.distance_to_target_evolution = distance_to_target_evolution


class ProjectedGradientDescent:
    """
    Implements projected gradient descent.
    """

    def __init__(self, epsilon, constraints):
        self.epsilon = epsilon
        self.cons = constraints
        self.f_star = optimize.minimize(lambda x: (
            x[0] - 2)**4 + (x[0] - 2*x[1])**2, (1, 1), method='SLSQP', constraints=self.cons)
        self.target = self.function(
            np.array([[self.f_star.x[0]], [self.f_star.x[1]]]))

    def function(self, x: np.array) -> float:
        return (x[0][0] - 2)**4 + (x[0][0] - 2*x[1][0])**2

    def gradient(self, x: np.array) -> np.array:
        return np.array([[4*(x[0][0] - 2)**3 + 2*(x[0][0] - 2*x[1][0])], [-4*(x[0][0] - 2*x[1][0])]])

    def projection(self, x: np.array) -> np.array:
        """
        Computes the projection of x on Q.
        """
        return min(1, 2/(sqrt(x[0]**2 + x[1]**2))) * x

    def stop(self, xk: np.array, epsilon: float) -> bool:
        """
        Checks if gradient(x_k) <= epsilon.
        """
        grad = self.gradient(xk)
        return sqrt(grad[0][0]**2 + grad[1][0]**2) <= epsilon

    def const_projected_gradient_descent(self, alpha: float = 0.1) -> Result:
        new_x, old_x, steps = self.projection(
            np.array([[1], [1]])), self.projection(np.array([[1], [1]])), 0
        gradient_norm_evolution = []
        function_evolution = []
        while not self.stop(new_x, self.epsilon):
            new_x = self.projection(old_x - alpha * self.gradient(old_x))
            old_x = new_x
            grad = self.gradient(new_x)
            gradient_norm_evolution.append(sqrt(grad[0][0]**2 + grad[1][0]**2))
            function_evolution.append(abs(self.function(new_x) - self.target))
            steps += 1
        return Result(old_x, steps, gradient_norm_evolution, function_evolution)

    def backtracking_projected_gradient_descent(self) -> Result:
        new_x, old_x, steps = self.projection(
            np.array([[1], [1]])), self.projection(np.array([[1], [1]])), 0
        gradient_norm_evolution = []
        function_evolution = []
        while not self.stop(new_x, self.epsilon):
            alpha = self._compute_step_size_with_backtracking(old_x)
            new_x = self.projection(old_x - alpha * self.gradient(old_x))
            old_x = new_x
            grad = self.gradient(new_x)
            gradient_norm_evolution.append(sqrt(grad[0][0]**2 + grad[1][0]**2))
            function_evolution.append(abs(self.function(new_x) - self.target))
            steps += 1
        return Result(old_x, steps, gradient_norm_evolution, function_evolution)

    def _compute_step_size_with_backtracking(self, old_x: np.array) -> float:
        c, ro = 0.5, 0.1
        alpha = 0.5
        new_x = self.projection(old_x - alpha * self.gradient(old_x))
        while self.function(new_x) > self.function(old_x) - (c/alpha) * ((new_x[0][0] - old_x[0][0])**2 - (new_x[1][0] - old_x[1][0])**2):
            alpha *= ro
            new_x = self.projection(old_x - alpha * self.gradient(old_x))
        return alpha


if __name__ == '__main__':
    solver = ProjectedGradientDescent(epsilon=10**-1, constraints=({'type': 'ineq', 'fun': lambda x: -
                                                                    1 * sqrt(x[0]**2 + x[1]**2) + 2}))
    const_results = solver.const_projected_gradient_descent()
    backtracking_results = solver.backtracking_projected_gradient_descent()

    plt.plot([i for i in range(1, const_results.steps + 1)],
             const_results.gradient_norm_evolution)
    plt.plot([i for i in range(1, backtracking_results.steps + 1)],
             backtracking_results.gradient_norm_evolution)
    plt.ylabel("||gradient(xk)||")
    plt.xlabel("k")
    plt.legend(["Constant Step(alpha = 0.1)", "Backtracking Step"])
    plt.show()

    plt.plot([i for i in range(1, const_results.steps + 1)],
             const_results.distance_to_target_evolution)
    plt.plot([i for i in range(1, backtracking_results.steps + 1)],
             backtracking_results.distance_to_target_evolution)
    plt.ylabel("||f(xk) - f*||")
    plt.xlabel("k")
    plt.legend(["Constant Step(alpha = 0.1)", "Backtracking Step"])
    plt.show()
