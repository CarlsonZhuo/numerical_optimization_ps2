import numpy as np
import numpy.linalg as la
from math import sqrt, cos, sin
import pdb

#######################################
######### Generate Data ###############
#######################################

def exponential(cx, cy, cz, C):
    M = np.matrix(C).reshape(3,3)
    theta = sqrt(cx**2+cy**2+cz**2)
    v1 = np.matrix([cx,cy,cz]).reshape(3,1)
    return v1*v1.T + cos(theta)/theta**2 * M*M + sin(theta)/theta*M


def c_to_exp(cx, cy, cz):
    C = np.array([0,-cz,cy,cz,0,-cx,-cy,cx,0])
    return exponential(cx, cy, cz, C)


data_size = 200
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
    samples_c = []
    for i in range(n):
        num_g = np.random.randint(0, 4)
        mean = means[num_g]
        cx, cy, cz = np.random.multivariate_normal(mean, var*(num_g+1))
        C = np.array([0,-cz,cy,cz,0,-cx,-cy,cx,0])
        sample = exponential(cx,cy,cz,C)
        sample = np.array(sample.reshape(1,9))[0]
        sample_c = np.array([cx,cy,cz])
        samples.append(sample)
        samples_c.append(sample_c)
    return np.array(samples), samples_c


samples, samples_c = GMMSamples(data_size)
samples = np.matrix(samples).T
#######################################
########### Build Model ###############
#######################################

# Convert the max problem into min problem
def loss(x):
    exp_x = c_to_exp(x[0,0],x[1,0],x[2,0]).reshape(9,1)
    result = - la.norm(exp_x-samples, axis=0) / (2*sigma**2)  # returns an array
    result = sum(np.exp(result))
    return - result

dC_dcx = np.matrix('0,0,0;0,0,-1;0,1,0')
dC_dcy = np.matrix('0,0,1;0,0,0;-1,0,0')
dC_dcz = np.matrix('0,-1,0;1,0,0;0,0,0')

def grad_f(x):
    exp_x = c_to_exp(x[0,0],x[1,0],x[2,0])
    exp_x = np.matrix(exp_x).reshape(3,3)
    vec_exp_x = np.matrix(exp_x).reshape(9,1)
    result = np.array([0.0,0.0,0.0])
    dexpC_dcx = (exp_x * dC_dcx).reshape(9,1)
    dexpC_dcy = (exp_x * dC_dcy).reshape(9,1)
    dexpC_dcz = (exp_x * dC_dcz).reshape(9,1)

    for i in range(data_size):
        vec_xi = samples[:,i]
        xi = vec_xi.reshape(3,3)
        common_scalar = np.exp(-la.norm(vec_exp_x - vec_xi) / (2*sigma**2))
        delta_x = common_scalar * 2 * np.multiply((vec_exp_x - vec_xi),(dexpC_dcx))
        delta_y = common_scalar * 2 * np.multiply((vec_exp_x - vec_xi),(dexpC_dcy))
        delta_z = common_scalar * 2 * np.multiply((vec_exp_x - vec_xi),(dexpC_dcz))
        result[0] += np.sum(delta_x)
        result[1] += np.sum(delta_y)
        result[2] += np.sum(delta_z)

    return - np.matrix(result).reshape(3,1)


def gradient_descend(eta, x_0, iter):
    x_i = x_0.reshape(3,1)
    for i in range(iter):
        x_i -= eta * grad_f(x_i)
    return x_i


def bfgs(x_0, iter):
    xk = np.matrix(x_0, copy=True).reshape(3,1)
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
    # grad_f_xk = np.matrix(grad_f(x_k)).reshape(3,1)
    benchmark = (grad_f(x_k).T * d_k)[0,0]
    while (loss(x_k) - loss(x_k+step_size*d_k) <
           -theta*step_size*benchmark):
        step_size *= beta
    return step_size

# Run Gradent Descent
x_ts_gd = []
loss_x_ts = []
for i in range(data_size):
    x_0 = np.matrix(samples_c[i], copy=True)
    x_t = gradient_descend(0.001, x_0, 1000)
    x_ts_gd.append(x_t)
    loss_x_ts.append(loss(x_t))
    print loss(x_t)

x_t_gd = x_ts_gd[np.argmin(loss_x_ts)]

# Run BFGS
x_ts_bfgs = []
for i in range(data_size):
    x_0 = np.array(samples_c[i], copy=True)
    x_t = bfgs(x_0, 3)
    x_ts_bfgs.append(x_t)
    print loss(x_t)

