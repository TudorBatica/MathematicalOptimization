import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from numpy import linalg as la
from scipy import optimize

# min_x (x1 - 2)^4 + (x1 - 2x2)^2
# s.l. x in Q = {||x|| <= 2}


class Result:
    """
    Data class that holds projected gradient descent results.
    """

    def __init__(self, solution: float = None, steps: int = 0,
                 gradient_norm_evolution: [float] = [], distance_to_target_evolution: [float] = []):
        self.solution = solution
        self.steps = steps
        self.gradient_norm_evolution = gradient_norm_evolution
        self.distance_to_target_evolution = distance_to_target_evolution


class ProjectedGradientDescent:
    """
    Implements projected gradient descent.
    """

    def __init__(self, constraints: dict, function, gradient, projection,
                 epsilon: float = 10**-1, alpha: float = 0.1):
        """
        Parameters
        ---
        epsilon: Accuracy threshold.\n
        costraints: Optimization constraints.\n
        function: f:R^2 -> R\n
        gradient: gradient(f)\n
        projection: projection of f on the space defined by the constraints.\n
        epsilon: Accuracy threshold.\n
        costraints: Optimization constraints.\n
        alpha: step size for the constant step solver or initial step size for the backtracking step solver
        """
        self.epsilon = epsilon
        self.cons = constraints
        self.function = function
        self.gradient = gradient
        self.projection = projection
        self.alpha = alpha
        __f_star = optimize.minimize(lambda x: (
            x[0] - 2)**4 + (x[0] - 2*x[1])**2, (1, 1), method='SLSQP', constraints=self.cons)
        self.target = self.function(
            np.array([[__f_star.x[0]], [__f_star.x[1]]]))

    def stop(self, xk: np.array, epsilon: float) -> bool:
        """
        Checks if gradient(x_k) <= epsilon.
        """
        grad = self.gradient(xk)
        return sqrt(grad[0][0]**2 + grad[1][0]**2) <= epsilon

    def const_projected_gradient_descent(self, alpha: float = 0.1) -> Result:
        return self._projected_gradient_descent(self._get_constant_step_size)

    def backtracking_projected_gradient_descent(self) -> Result:
        return self._projected_gradient_descent(self._get_step_size_with_backtracking)

    def _projected_gradient_descent(self, get_step_size_function) -> Result:
        new_x, old_x, steps = self.projection(
            np.array([[1], [1]])), self.projection(np.array([[1], [1]])), 0
        gradient_norm_evolution = []
        function_evolution = []
        while not self.stop(new_x, self.epsilon):
            alpha = get_step_size_function(old_x)
            new_x = self.projection(old_x - alpha * self.gradient(old_x))
            grad = self.gradient(new_x)
            gradient_norm_evolution.append(sqrt(grad[0][0]**2 + grad[1][0]**2))
            function_evolution.append(abs(self.function(new_x) - self.target))
            old_x = new_x
            steps += 1
        return Result(old_x, steps, gradient_norm_evolution, function_evolution)

    def _get_step_size_with_backtracking(self, old_x: np.array) -> float:
        c, ro = 0.5, 0.1
        alpha = 0.5
        new_x = self.projection(old_x - alpha * self.gradient(old_x))
        while self.function(new_x) > self.function(old_x) - (c/alpha) * ((new_x[0][0] - old_x[0][0])**2 - (new_x[1][0] - old_x[1][0])**2):
            alpha *= ro
            new_x = self.projection(old_x - alpha * self.gradient(old_x))
        return alpha

    def _get_constant_step_size(self, old_x: np.array) -> float:
        return self.alpha


if __name__ == '__main__':
    constraints = ({'type': 'ineq', 'fun': lambda x: -
                   1 * sqrt(x[0]**2 + x[1]**2) + 2})

    def function(x): return (
        x[0][0] - 2)**4 + (x[0][0] - 2*x[1][0])**2

    def gradient(x): return np.array(
        [[4*(x[0][0] - 2)**3 + 2*(x[0][0] - 2*x[1][0])],
         [-4*(x[0][0] - 2*x[1][0])]])

    def projection(x): return min(1, 2/(sqrt(x[0]**2 + x[1]**2))) * x

    solver = ProjectedGradientDescent(function=function, gradient=gradient, projection=projection,
                                      constraints=constraints)

    const_results = solver.const_projected_gradient_descent()
    backtracking_results = solver.backtracking_projected_gradient_descent()
    
    # plot results
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
