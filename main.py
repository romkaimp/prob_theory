from functools import reduce
from scipy.integrate import quad
from math import pi, exp
from numpy import array
import asyncio


#class Functions():
#    n = None
#    m = None
#    p = None
#
#    def __init__(self, n: int, m: int, p: float = 0):
#        self.n = n
#        self.m = m
#        self.p = p
#
#
#    class CashSolution():
#        cash: list = []
#        f = Relations()
#
#        class Relations():
#            def __init__(self):
#                pass
#
#            def
#
#        def counting(expression: str):
#
#
#
#    def factorial(self):
#        k = 1
#        for i in range(1, self.n + 1):
#            k *= i
#        return k
#
#    def p_(self):
#        return self.factorial()
#
#    def p_r(n, *args):
#        j = [factorial(x) for x in args]
#        multiply_res = reduce((lambda x, y: x * y), j)
#        return factorial(n) / multiply_res
#
#    def a_(n, m):
#        return factorial(n) / factorial(n - m)
#
#    def a_r(n, m):
#        return pow(n, m)
#
#    def c_(n, m):
#        return a_(n, m) / factorial(m)
#
#    def c_r(n, m):
#        res = c_(m + n - 1, m - 1)
#        if res < 1:
#            return 1
#        return res
#
#    def bernulli(n, k, q1, q2: float = 0):
#        q2 = 1 - q1
#        return c_(n, k) * pow(q1, k) * pow(q2, n - k)
#
#    def muavr_laplace(n, m, p, pi=pi):
#        q = 1 - p
#        return (1 / pow(2 * pi * n * p * q, 1 / 2)) * exp(-(pow(m - n * p, 2)) / (2 * n * p * q))
#
#    def muavr_integral(n, k1, k2, p, pi=pi):
#        q = 1 - p
#
#        def integrand(x):
#            return exp(-pow(x, 2) / 2)
#
#        return 1 / (pow(2 * pi, 1 / 2)) * quad(integrand, (k1 - n * p) / pow(n * p * q, 1 / 2),
#                                               (k2 - n * p) / pow(n * p * q, 1 / 2))


#"""4 задача"""
#print("1:", bernulli(6, 2, 0.2))
#print("2:", bernulli(6, 2, 0.2)*bernulli(7, 3, 0.3))
#
#C = bernulli(6, 2, 0.2)*bernulli(7, 0, 0.3)+bernulli(6, 1, 0.2)*bernulli(7, 1, 0.3)+bernulli(7, 2, 0.3)*bernulli(6, 0, 0.2)
#
#print("3:", C)
#P = 0
#for i in range(1, 6+1):
#    for j in range(0, i):
#        P += bernulli(6, i, 0.2)*bernulli(7, j, 0.3)
#
#print("4:", P)
#print("5:", (bernulli(6, 2, 0.2)*bernulli(7, 0, 0.3)+bernulli(6, 0, 0.2)*bernulli(7, 2, 0.3))/C)
#
#P2 = 0
#for i in range(3, 6+1):
#    for j in range(0, i):
#        P2 += bernulli(6, i, 0.2)*bernulli(7, j, 0.3)
#print("6:", P2/P)

class Counter(object):
    """singleton"""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self, cls):
        pass


class Adapter:
    pass


class Mediator:
    pass


class Algoth(object):
    attrs = {}
    n = None
    priority = 10
    hist: int = None

    def __new__(cls, *args, **kwargs):
        if args[0] in cls.attrs:
            return cls.attrs[args[0]]
        cls.attrs[args[0]] = object.__new__(cls)
        return cls.attrs[args[0]]

    def __init__(self, n):
        self.n = n

    def __repr__(self):
        if self.hist:
            return str(self.hist)
        else:
            return str(self.__call__())

    def __call__(self, *args, **kwargs):
        self.hist = self.factorial(self.n)
        return self.hist

    def factorial(self, n):
        k = 1
        for i in range(1, self.n + 1):
            k *= i
        return k


class A(Algoth):
    m = 0
    hist: int = None

    def __init__(self, n, m):
        super(A, self).__init__(n)
        self.m = m

    def __call__(self, *args, **kwargs):
        k = int(super(A, self).__call__() / Algoth(self.n - self.m)())
        return k


class P(Algoth):
    hist: int = None

    def __init__(self, n):
        super(P, self).__init__(n)


class Pr(Algoth):
    hist: int = None
    args = []

    def __init__(self, n, *args):
        super(Pr, self).__init__(n)
        self.args = [Algoth(x) for x in args]

    def __call__(self, *args, **kwargs):

        k = super(Pr, self).__call__()
        k /= reduce((lambda x, y: x() * y()), self.args)
        self.hist = int(k)
        return int(k)

    def __add__(self, other):
        return


p = A(10, 2)

g = P(5)
k = P(5)
f = P(6)
print(id(P(5)), id(k), k, f, id(f))
print(p())
print(p.__dict__)
