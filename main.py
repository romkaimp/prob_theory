from functools import reduce
from scipy.integrate import quad
from math import pi, exp
from numpy import array



class Functions():
    n = None
    m = None
    p = None

    def __init__(self, n: int, m: int, p: float = 0):
        self.n = n
        self.m = m
        self.p = p


    class CashSolution():
        cash: list = []
        f = Relations()

        class Relations():
            def __init__(self):
                pass

            def

        def counting(expression: str):



    def factorial(self, n: int=-1):
        if self.n == None:
            if n != -1:
                self.n = n
        if n != None:
        k = 1
        for i in range(1, n + 1):
            k *= i
        return k

    def p_(n):
        return factorial(n)

    def p_r(n, *args):
        j = [factorial(x) for x in args]
        multiply_res = reduce((lambda x, y: x * y), j)
        return factorial(n) / multiply_res

    def a_(n, m):
        return factorial(n) / factorial(n - m)

    def a_r(n, m):
        return pow(n, m)

    def c_(n, m):
        return a_(n, m) / factorial(m)

    def c_r(n, m):
        res = c_(m + n - 1, m - 1)
        if res < 1:
            return 1
        return res

    def bernulli(n, k, q1, q2: float = 0):
        q2 = 1 - q1
        return c_(n, k) * pow(q1, k) * pow(q2, n - k)

    def muavr_laplace(n, m, p, pi=pi):
        q = 1 - p
        return (1 / pow(2 * pi * n * p * q, 1 / 2)) * exp(-(pow(m - n * p, 2)) / (2 * n * p * q))

    def muavr_integral(n, k1, k2, p, pi=pi):
        q = 1 - p

        def integrand(x):
            return exp(-pow(x, 2) / 2)

        return 1 / (pow(2 * pi, 1 / 2)) * quad(integrand, (k1 - n * p) / pow(n * p * q, 1 / 2),
                                               (k2 - n * p) / pow(n * p * q, 1 / 2))


def factorial(n):
    k = 1
    for i in range(1, n + 1):
        k *= i
    return k


def p_(n):
    return factorial(n)


def p_r(n, *args):
    j = [factorial(x) for x in args]
    multiply_res = reduce((lambda x, y: x*y), j)
    return factorial(n)/multiply_res


def a_(n, m):
    return factorial(n)/factorial(n-m)


def a_r(n, m):
    return pow(n, m)


def c_(n, m):
    return a_(n, m)/factorial(m)


def c_r(n, m):
    res = c_(m + n - 1, m - 1)
    if res < 1:
        return 1
    return res


def bernulli(n, k, q1, q2: float = 0):
    q2 = 1 - q1
    return c_(n, k)*pow(q1, k)*pow(q2, n-k)


def muavr_laplace(n, m, p, pi=pi):
    q = 1 - p
    return (1/pow(2*pi*n*p*q, 1/2))*exp(-(pow(m - n*p, 2))/(2*n*p*q))

def muavr_integral(n, k1, k2, p, pi=pi):
    q = 1 - p
    def integrand(x):
        return exp(-pow(x, 2)/2)
    return 1/(pow(2*pi, 1/2)) * quad(integrand, (k1- n*p)/pow(n*p*q, 1/2), (k2 - n*p)/pow(n*p*q, 1/2))


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