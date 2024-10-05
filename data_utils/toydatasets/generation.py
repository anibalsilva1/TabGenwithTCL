import numpy as np

from .decision_boundaries import *
from sklearn import datasets


def generate_moon_data(n_class1, n_class2, noise=0.05):
    '''
    Generates 2D artificial data moon shaped for binary classification.

    param n_class1 int: number of points for class 1,
    param n_class2 int: number of points for class 2,
    param noise float: standard deviation addition to data.
    '''

    X, y = datasets.make_moons(n_samples=(n_class1, n_class2, ),
                               noise=noise)

    return X, y


def generate_xor_data(N):
    '''
    Generates XOR dataset.
    
    param N int: number of points.
    '''

    X = np.random.random(N*2)
    X = X.reshape((N,2))

    y = np.array(list(map(lambda x: xor(x), X)))

    return X, y

def generate_kite_data(N, noise_std):
    '''
    Generates data in a form of a parrot.

    param N int: number of points,
    param noise_std float: standard deviation addition to data.
    '''

    x_coord = np.random.uniform(-1,1,N) + noise_std*np.random.randn(N)
    y_coord = np.random.uniform(-1,1,N) + noise_std*np.random.randn(N)
    
    X = np.stack([x_coord, y_coord], axis=1)
    y = np.array(list(map(lambda x: kite(x), X)))

    # remove outer label
    #X = X[y!=4]
    #y = y[y!=4]


    return X, y

def generate_hourglass_data(N, noise_std):
    '''
    Generates data in the form of a hourglass.

    param N int: number of points,
    param noise_std float: standard deviation addition to data.
    '''

    x_coord = np.random.uniform(-1,1,N) + noise_std*np.random.randn(N)
    y_coord = np.random.uniform(-1,1,N) + noise_std*np.random.randn(N)
    
    X = np.stack([x_coord, y_coord], axis=1)
    y = np.array(list(map(lambda x: hourglass(x), X)))

    return X, y


def generate_sin_wave_data(N, noise_std, k=1):
    '''
    Generates data in a form of a wave.

    param N int: number of points,
    param k int: number of periods.
    param noise_std float: standard deviation addition to data.
    '''

    x_coord = np.random.uniform(0, 2*k*math.pi, N) + noise_std*np.random.randn(N)
    y_coord = np.random.uniform(-1, 1, N) + noise_std*np.random.randn(N)
    
    X = np.stack([x_coord, y_coord], axis=1)
    y = np.array(list(map(lambda x: sin_wave(x), X)))

    return X, y

def generate_flower_data(N, a, b, radius, noise_std, two_class=True):
    '''
    Generates data of a flower with 4 petals and a circle in the center.

    The flower petals labels are assigned given ellipses decision boundaries. Each petals has its own label.

    The ellipses along the x axis have the decision equation: ((x - u)/a)^2 + (y/b)^2 < 1
    The ellipses along the y axis have the decision equation (x/a)^2 + ((y-v)/b)^2 < 1

    param N int: number of points,
    param noise_std float: standard deviation addition to data,
    param a float: semi-major axis of an ellipse,
    param b float: semi-minor axis of an ellipse,
    param r float: radius of centered circle.
    '''

    x_coord = np.random.uniform(-2*(a+radius),2*(a+radius),N) + noise_std*np.random.randn(N)
    y_coord = np.random.uniform(-2*(a+radius),2*(a+radius),N) + noise_std*np.random.randn(N)
    
    X = np.stack([x_coord, y_coord], axis=1)
    y = np.array(list(map(lambda x: flower(x, a, b, radius, two_class), X)))
    # remove background
    #X = X[y!=5]
    #y = y[y!=5]

    return X, y


def generate_circle_data(N, radius, noise_std):
    '''
    Generates data where one of the classes is encompassed in a circle.

    param N int: number of points,
    param radius int: radius of circle.
    param noise_std float: standard deviation addition to data.
    '''

    x_coord = np.random.uniform(-(3*radius), (3*radius), N) + noise_std*np.random.randn(N)
    y_coord = np.random.uniform(-(3*radius), (3*radius), N) + noise_std*np.random.randn(N)

    X = np.stack([x_coord, y_coord], axis=1)
    y = np.array(list(map(lambda x: circles(x, radius), X)))

    return X, y
