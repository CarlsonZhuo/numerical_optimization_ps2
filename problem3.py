import numpy as np
import numpy.linalg as la
import pdb

#######################################
######### Generate Data ###############
#######################################

data_size = 20
means = []
means.append(np.array([2,2,0]))
means.append(np.array([-2,-2,0]))
means.append(np.array([2,-2,2]))
means.append(np.array([-2,2,-2]))

var = np.array([1,0,0,0,1,0,0,0,1]).reshape(3,3)
sigma = 1

# Sample from 4 Gaussians
# with variances all being 1
# and means being different
def GMMSamples(n):
    samples = []
    for i in range(n):
        num_g = np.random.randint(0, 4)
        mean = means[num_g]
        samples.append(np.random.multivariate_normal(mean, var*(num_g+1)))
    return np.array(samples)


samples = GMMSamples(data_size)
samples = np.matrix(samples).T


#######################################
########### Build Model ###############
#######################################

# Convert the max problem into min problem
def loss(x):
    result = - la.norm(x-samples, axis=0) / (2*sigma**2)  # returns an array
    result = sum(np.exp(result))
    return - result

def grad_f(x):
    result = - la.norm(x-samples, axis=0) / (2*sigma**2)  # returns an array
    result = np.matrix(result).T
    result = (x-samples) * (-np.exp(result)/sigma**2)
    return - result

def gradient_descend(eta, x_0, iter):
    x_i = x_0
    for i in range(iter):
        x_i -= eta * grad_f(x_i)
    return x_i

def bfgs(x_0, iter):
    xk = np.matrix(x_0, copy=True)
    Bk = np.matrix(np.identity(3))
    for i in range(iter):
        grad_f_xk = grad_f(xk)
        pk = - la.inv(Bk) * grad_f_xk
        step_size = line_search(xk, pk)
        if step_size < 0.000000001:
            return xk
            # To avoid devide by zero error
            # Extreme non-convex point encountered
        sk = step_size * pk
        x_kp1 = xk + sk
        yk = grad_f(x_kp1) - grad_f_xk

        Bk += (yk*yk.T)/(yk.T*sk) - Bk*sk*sk.T*Bk/(sk.T*Bk*sk)
        xk = x_kp1
    return xk


# The line search algorithm for the second order algo
def line_search(x_k, d_k):
    # The linear approximation at x-x_k
    beta = 0.5
    step_size = 1
    theta = 0.0001
    benchmark = (grad_f(x_k).T * d_k)[0,0]
    while (loss(x_k) - loss(x_k+step_size*d_k) <
           -theta*step_size*benchmark):
        step_size *= beta
    return step_size

# Run Gradient Descent
x_ts_gd = []
loss_x_ts = []
for i in range(data_size):
    x_0 = np.matrix(samples[:,i], copy=True)
    x_t = gradient_descend(0.001, x_0, 1000)
    x_ts_gd.append(x_t)
    loss_x_ts.append(loss(x_t))
    print loss(x_t)

x_t_gd = x_ts_gd[np.argmin(loss_x_ts)]

# Run BFGS
x_ts_bfgs = []
for i in range(data_size):
    x_0 = np.array(samples[:,i])
    x_t = bfgs(x_0, 3)
    x_ts_bfgs.append(x_t)
    print loss(x_t)

