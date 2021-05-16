import numpy as np

np.set_printoptions(suppress=True)

def load_cleveland_dataset():
    """ 
    The Cleveland Heart Disease Dataset. 0 - no heart disease/ 1-4 => some heart disease. 
    The records get relabeled as -1 - no heart disease / 1 - some heart disease.
    """
    with open('pegasos/dataset.txt', 'r') as f:
        x = []
        y = []
        lines = f.read().splitlines()
        for line in lines:
            data = line.split(',')
            data = [float(d) for d in data]
            x.append(data[0:-1])
            y.append(-1 if not data[-1] else 1)
    
    train_data_size = int(len(x) * 0.7)
    S = [(data, label) for data, label in zip(x[0:train_data_size], y[0:train_data_size])]
    
    return S, x[train_data_size:], y[train_data_size:]

def subset(S, k):
    """
    S - initial set
    k - subset length
    """
    return S[:k]

"""
def inner_product(vector1, vector2):
    product = 0
    for v1, v2 in zip(vector1, vector2):
        product += v1 * v2
    return product
"""

def w_t_and_a_half(w, lmbda, At_plus, learning_rate, k):
    aux = w * (1 - learning_rate*lmbda)
    aux_sum = np.zeros(13,)
    for record in At_plus:
        aux_sum += record[1] * np.array(record[0])
    aux_sum *= learning_rate / k

    return aux + aux_sum

def next_w_t(w_t_half, lmbda):
    w_t_half_norm = np.linalg.norm(w_t_half)
    aux_min = min(1, (1/lmbda)/w_t_half_norm)

    return aux_min * w_t_half


def pegasos(S, lmbda, T, k):
    """ 
    Pegasos(non-kernel) algorithm. 
    S: set of training data
    lambda: ?
    T: no of iterations
    k: length of subset used in algorithm
    """
    w = np.ones(13,)
    for t in range(1, T + 1):
        At = subset(S, k)
        At_plus = np.array([record for record in At if record[1] * np.dot(w, record[0]) < 1], dtype=object)
        learning_rate = 1 / (lmbda * t)
        w_t_half = w_t_and_a_half(w, lmbda, At_plus, learning_rate, k)
        w = next_w_t(w_t_half, lmbda)
    
    return w

def test_nonkernel(w, X_test, y_test):
    total = 0
    correct = 0
    for (x,yi) in zip(X_test, y_test):
        pred = np.dot(w, x)
        if yi * pred > 0:
            correct += 1
        total += 1
    return correct, total

S, x_test, y_test = load_cleveland_dataset()
w = pegasos(S, lmbda = 1, T = 100, k = 75)
print(test_nonkernel(w, x_test, y_test))


def K(x1, x2):
    """
    The RBF Kernel
    """
    return np.exp(-1*np.linalg.norm(x1-x2)**2) 

def pegasos_kernel(S, T, lmbda):
    al = np.zeros(len(S))
    idx = np.random.permutation(len(S))
    #print(idx)
    t = 0
    for i in idx:
        x, yi = np.array(S[i][0]), S[i][1]
        s = 0
        for j in range(len(S)):
            s += al[j]*S[j][1] * K(x,np.array(S[j][0]))
        if yi*(1/lmbda)*s < 1:
            al[i] = al[i] + 1
        if t >= T:
            break
        else:
            t += 1
        print("Iteration of Kernel Training: ", t)
    return al

def test_kernel(al, X_test, y_test, S, T, lmbda):
    X_train = np.array([record[0] for record in S])
    y_train = [record[1] for record in S]
    total = 0
    correct = 0
    t = 0
    for (x,yi) in zip(X_test, y_test):
        s = 0
        for j in range(len(X_train)):
            s += al[j]*y_train[j]*K(x,X_train[j])
        if yi*(1/lmbda)*s < 1:
            correct += 1
        total += 1
        if t >= T:
            break
        else:
            t += 1
        print("Testing iteration: ", t)
    return correct, total

al = pegasos_kernel(S, 10, 1)
print(al)
print(test_kernel(al, x_test, y_test, S, 10, 1))