from abc import ABC, abstractmethod
import numpy as np

class Tableau(ABC):
    """Template for Butcher Tableaux
    
    Butcher tableaux are a descroption of a particular Runge-Kutta method.
    They specify all the coefficients used for computing intermediate
    stages, which are used to compute the final prediction.

    Attributes:
        a (np.array): a square matrix of spatial interpolation coefficients
        b (np.array): a 1D matrix of final interpolation coefficients
        c (np.array): a 1D matrix of temporal interpolation coefficients
        order (int): accuracy order of method
        s (int): number of stages
        is_explicit (bool): whether the method is explicit or implicit

    Note: all 
    """
    def __init__(self):
        if not isinstance(self.a,np.ndarray):
            raise TypeError('self.a must be of type np.ndarray')
        if not isinstance(self.b,np.ndarray):
            raise TypeError('self.b must be of type np.ndarray')
        if not isinstance(self.c,np.ndarray):
            raise TypeError('self.c must be of type np.ndarray')
        if self.a.shape[0] != self.a.shape[1]:
            raise ValueError('self.a must be square')
        
        if self.a.shape[0] != self.b.shape[0]:
            raise ValueError('self.a.shape[0] must equal self.b.shape[0]')
        if self.b.shape[0] != self.c.shape[0]:
            raise ValueError('self.b.shape[0] must equal self.c.shape[0]')

    def __str__(self):
        return "Unnamed tableau"

    @property
    @abstractmethod
    def a(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def b(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def c(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def order(self):
        raise NotImplementedError

    @property
    def s(self):
        return self.a.shape[0]

    @property
    def is_explicit(self):
        return np.allclose(self.a, np.tril(self.a, k=-1))

    @property
    def is_adaptive(self):
        return hasattr(self, 'blo')


class ForwardEuler(Tableau):
    order = 1
    a = np.array([[0.]])
    b = np.array([1.])
    c = np.array([1.])

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "Forward Euler: order {:d}".format(self.order)


class ExplicitMidpoint(Tableau):
    order = 2
    a = np.array([[0., 0.],
                  [1/2, 0.]])
    b = np.array([0., 1.])
    c = np.array([0., 1/2])

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "ERK: order {:d}".format(self.order)


class ExplicitRungeKutta3(Tableau):
    order = 3
    a = np.array([[0., 0.,   0.],
                  [1/2, 0.,   0.],
                  [-1., 2,   0.]])
    b = np.array([1/6, 2/3, 1/6])
    c = np.array([0., 1/2, 1.])

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "ERK: order {:d}".format(self.order)


class ExplicitRungeKutta4(Tableau):
    order = 4
    a = np.array([[0., 0.,   0., 0.],
                  [1/2, 0.,   0., 0.],
                  [0., 1/2,   0., 0.],
                  [0., 0.,   1., 0.]])
    b = np.array([1/6, 1/3, 1/3, 1/6])
    c = np.array([0., 1/2, 1/2, 1.])

    def __init__(self):
        super().__init__()

    def __str__(self):
        return "ERK: order {:d}".format(self.order)

class Dopri45(Tableau):
    order = 4
    a = np.array([[0., 0., 0., 0., 0., 0., 0.],
                  [1/5, 0., 0., 0., 0., 0., 0.],
                  [3/40, 9/40, 0., 0., 0., 0., 0.],
                  [44/45, -56/15, 32/9, 0., 0., 0., 0.],
                  [19372/6561, -25360/2187, 64448/6561, -212/729, 0., 0., 0.],
                  [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
                  [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]]) 
    b = np.array([35/384, 0., 500/1113, 125/192, -2187/6784, 11/84, 0])
    blo = np.array([5179/57600, 0., 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
    c = np.array([0., 1/5, 3/10, 4/5, 8/9, 1., 1.])

    def __init__(self, atol=1e-5, rtol=1e-5):
        super().__init__()
        self.atol = atol
        self.rtol = rtol

    def __str__(self):
        return "DoPri 4(5)"



        

        

















