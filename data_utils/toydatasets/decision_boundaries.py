import math

def xor(x):
    '''
    Generate XOR instances. Taken from ML classes.
    
    param x numpy array: 2D features.
    '''

    if (x[0]>0.5 and x[1]>0.5) or (x[0]<=0.5 and x[1]<=0.5):
        return 1
    else:
        return 0

def kite(x):
    '''
    Gives classes in order to produce a shape of a kite.
    '''
    if x[1] <= -x[0]+1 and x[0] >= 0 and x[1] >= 0:
        return 0
    elif x[1] >= -1+x[0] and x[0] >= 0 and x[1] <= 0:
        return 1
    elif x[1] >= -1-x[0] and x[0] <= 0 and x[1] <= 0:
        return 2
    elif x[1] <= x[0]+1 and x[0] <= 0 and x[1] >= 0:
        return 3
    else:
        return 4


def hourglass(x):
    '''
    Gives classes in order to produce a shape of an hourglass.
    '''
    if x[1] <= - x[0]**2 or x[1] >= x[0]**2:
        return 0
    else:
        return 1


def sin_wave(x):
    '''
    Gives classes in order to produce classes below and above a sin function.
    '''
    if x[1] < math.sin(x[0]):
        return 0
    else:
        return 1


def flower(x, a, b, radius, two_class = True):
    '''
    Generates a flower with 4 petals and a circle in the center and attributes classes to each petal.

    The flower petals labels are assigned given ellipses decision boundaries. 
    The ellipses along the x axis have the decision equation: ((x - u)/a)^2 + (y/b)^2 < 1
    The ellipses along the y axis have the decision equation (x/a)^2 + ((y-v)/b)^2 < 1

    param a float: semi-major axis of an ellipse,
    param b float: semi-minor axis of an ellipse,
    param r float: radius of centered circle.
    param two_cass bol: if True, all petals have the same label.
    '''

    u = a+radius # center of the ellipses along x axis (with y = 0)
    v = a+radius # center of the ellipses along y axis (with x = 0)

    if two_class:

        if x[0]**2 + x[1]**2 < radius:
            return 0
        
        elif ((x[0] - u)/a)**2 + (x[1]/b)**2 < radius:
            return 1

        elif ((x[0] + u)/a)**2 + (x[1]/b)**2 < radius:
            return 1

        elif (x[0]/b)**2 + ((x[1] - v)/a)**2 < radius:
            return 1

        elif (x[0]/b)**2 + ((x[1] + v)/a)**2 < radius:
            return 1

        else:
            return 2

    else:

        if x[0]**2 + x[1]**2 <= radius:
            return 0

        if ((x[0] - u)/a)**2 + (x[1]/b)**2 <= radius:
            return 1

        elif ((x[0] + u)/a)**2 + (x[1]/b)**2 <= radius:
            return 2

        elif (x[0]/b)**2 + ((x[1] - v)/a)**2 <= radius:
            return 3

        elif (x[0]/b)**2 + ((x[1] + v)/a)**2 <= radius:
            return 4

        else:
            return 5
        

def circles(x, radius):
    '''
    Gives a class to a point if it is outside or inside a circle.
    '''

    if x[0]**2 + x[1]**2 < radius**2:
        return 0
    elif x[0]**2 + x[1]**2 >= radius**2 and x[0]**2 + x[1]**2 < (2*radius)**2:
        return 1
    elif x[0]**2 + x[1]**2 >= (2*radius)**2 and x[0]**2 + x[1]**2 < (3*radius)**2:
        return 2
    else:
        return 3