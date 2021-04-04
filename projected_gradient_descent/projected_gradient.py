import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from numpy import linalg as la
from scipy import optimize

## min_x (x1 - 2)^4 + (x1 - 2x2)^2 
## s.l. x in Q = {||x|| <= 2}

epsilon = 10**-3

starting_value = np.array([[1], [1]])

def function(x: np.array) -> float:
    return (x[0][0] - 2)**4 + (x[0][0] - 2*x[1][0])**2

def gradient(x: np.array) -> np.array:
    return np.array([[4*(x[0][0] - 2)**3 + 2*(x[0][0] - 2*x[1][0])], [-4*(x[0][0] - 2*x[1][0])]])

def projection(x: np.array) -> np.array:
    return min(1, 2/(sqrt(x[0]**2 + x[1]**2))) * x

def stop(fun_value: float, target_value: float, epsilon: float) -> bool:
    return abs(fun_value - target_value) <= epsilon

def const_projected_gradient_descent():
    cons = ({'type': 'ineq', 'fun': lambda x:  -1 * sqrt(x[0]**2 + x[1]**2) + 2})
    f_star = optimize.minimize(lambda x: (x[0] - 2)**4 + (x[0] - 2*x[1])**2, (1, 1), method='SLSQP', constraints=cons).x

    print(f_star)
    alpha = 0.1
    new_x, old_x, steps = np.array([[1], [1]]), np.array([[1], [1]]), 0
    while not stop(function(new_x), function(np.array([[f_star[0]], [f_star[1]]])), epsilon):
        new_x = projection(old_x - alpha * gradient(old_x))
        old_x = new_x
        steps += 1
    return old_x, steps

def backtracking_projected_gradient_descent():
    cons = ({'type': 'ineq', 'fun': lambda x:  -1 * sqrt(x[0]**2 + x[1]**2) + 2})
    f_star = optimize.minimize(lambda x: (x[0] - 2)**4 + (x[0] - 2*x[1])**2, (1, 1), method='SLSQP', constraints=cons).x
    
    new_x, old_x, steps = np.array([[1], [1]]), np.array([[1], [1]]), 0
    c, ro = 0.5, 0.5
    while not stop(function(new_x), function(np.array([[f_star[0]], [f_star[1]]])), epsilon):
        alpha = 1
        new_x = projection(old_x - alpha * gradient(old_x))
        while function(new_x) > function(old_x) - (c/alpha) * ((new_x[0][0] - old_x[0][0])**2 - (new_x[1][0] - old_x[1][0])**2):
            alpha *= ro
            new_x = projection(old_x - alpha * gradient(old_x))
            print(alpha)
        new_x = projection(old_x - alpha * gradient(old_x))
        old_x = new_x
        steps += 1
    return old_x, steps

#print(function(np.array([[1.69], [0.84]])))
#print(gradient(np.array([[2], [2]])))


print(f'constant step:\n{const_projected_gradient_descent()}')
print(f'backtracked step:\n{backtracking_projected_gradient_descent()}')


"""
## Date: Q, q
Q = np.array([[1,1],[1,1.02]])
q = np.array([[2],[2]])
a = np.array(np.random.rand(2,1))
sol = -la.inv(Q)@q
b = a.T@sol - np.array(np.random.rand(1,1))
f = lambda x: 0.5*x.T@Q@x + q.T@x
Proj = lambda x: x - (np.maximum(0,a.T@x - b)/(la.norm(a)**2))*a

#### Rezolvare folosing CVXPY
n = Q.shape[0]
z = cp.Variable(n)
objective = cp.Minimize(0.5*cp.quad_form(z,Q) + q.T@z)
constraints = [a.T@z<=b]
prob = cp.Problem(objective, constraints)
result = prob.solve(solver='CVXOPT')
sol = z.value        


k = 0
eps = 10e-4
x00 = np.array(70*np.random.rand(2,1))
x0 = Proj(x00)

### Constanta Lipschitz a gradientului
Lips = np.max(la.eigvals(Q))
alpha = 1/Lips
print(alpha)
all_x_i = [x0[0]]
all_y_i = [x0[1]]
all_f_i = [f(x0)]
x_old = x0
x = Proj(x_old - alpha*(Q@x_old + q))
all_x_i.append(x[0])
all_y_i.append(x[1])
all_f_i.append(f(x))
criteriu_stop = la.norm(x - x_old)

while (criteriu_stop > eps**2):
    x_old = x
    
    ## Pas gradient
    grad = Q@x+q
    y = x - alpha*grad
    
    ## Pas proiectie
    x = Proj(y)
    
    #print(a.T@x-b)
    criteriu_stop = la.norm(x - x_old)
    k = k+1

    all_x_i.append(x[0])
    all_y_i.append(x[1])
    all_f_i.append(f(x))

solMGP = x      

print("#######################")
print("")
print("Solutie gasita de MG: ",'(',solMGP[0],solMGP[1],')')
print("Solutie optima: ",'(',sol[0],sol[1],')')
print("Valoare optima: ",f(sol))
print("")
print("#######################")   
#### Plot ####


x, y = np.mgrid[-10:10:0.1, -10:10:0.1]
x = x.T
y = y.T

fig = plt.figure(1, figsize=(4, 4))
plt.clf()
plt.axes([0, 0, 1, 1])

fcont = 0.5*Q[0,0]*(x**2) + 0.5*Q[1,1]*(y**2) + Q[1,0]*x*y + q[0]*x+ q[1]*y
contours = plt.contour(fcont, extent=[-10, 10, -10, 10],
                    cmap=plt.cm.gnuplot)



#Etichete pentru multimile izonivel

plt.clabel(contours, inline=1,
                fmt='%1.1f', fontsize=10)

plt.plot(all_x_i, all_y_i, 'b-', linewidth=2)
plt.plot(all_x_i, all_y_i, 'k+')
plt.plot(sol[0], sol[1], 'rx', markersize = 14)





fig.savefig('MGP_quad_cs.pdf')
"""

