from functools import reduce
from scipy.integrate import quad
from math import pi, exp, sqrt
from matplotlib.pyplot import plot, show, grid, axvline, xticks, yticks, axhline
from typing import Callable
from PIL import Image
import time
from datetime import datetime

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
    n: int = None
    priority = 10
    hist: int = None

    def __new__(cls, *args, **kwargs):
        #print(args)
        if args in cls.attrs:
            return cls.attrs[args]
        cls.attrs[args] = object.__new__(cls)
        return cls.attrs[args]

    def __init__(self, n: int):
        self.n = n

    def __repr__(self):
        return str(self.n)
        #if self.hist:
        #    return str(self.hist)
        #else:
        #    return str(self.__call__())

    def __str__(self):
        return str(self.n)

    def __call__(self, *args, **kwargs) -> int:
        if len(args) == 0:
            self.hist = self.factorial(*args)
            k = self.hist
        else:
            k = self.factorial(*args)
        return k

    def factorial(self, start_from: int | Callable = 0) -> int:
        summ = 1
        if isinstance(start_from, Algoth):
            k = start_from.n + 1
        else:
            k = start_from + 1
        for i in range(k, self.n + 1):
            summ *= i
        #print(f"{self.n, start_from}!={summ}")
        return summ

    def __add__(self, other: int | Callable) -> int:
        if isinstance(other, Algoth):
            return self.__call__() + other.__call__()
        else:
            return self.__call__() + other

    def __mul__(self, other: int | float | Callable) -> int:
        if isinstance(other, Algoth):
            return self.__call__() * other.__call__()
        else:
            return self.__call__() * other

    def __rmul__(self, other: int | float | Callable) -> int:
        if isinstance(other, Algoth):
            return self.__call__() * other.__call__()
        else:
            return self.__call__() * other

    def __int__(self):
        return self.__call__()

    def __truediv__(self, other):
        if isinstance(other, Algoth):
            r = other.__call__()
            if r == 0:
                raise ZeroDivisionError
            return self.__call__() // r
        else:
            if other == 0:
                raise ZeroDivisionError
            return self.__call__() / other

    def __rtruediv__(self, other):
        if isinstance(other, Algoth):
            return other.__call__() / self.__call__()
        else:
            return other / self.__call__()

    def __lt__(self, other):
        return self.__call__() < other

    def __le__(self, other):
        return self.__call__() <= other

    def __eq__(self, other):
        return self.__call__() == other

    def __ne__(self, other):
        return self.__call__() != other

    def __gt__(self, other):
        return self.__call__() > other

    def __ge__(self, other):
        return self.__call__() >= other

class A(Algoth):
    m = 0
    attrs = {}
    hist: int = None

    def __new__(cls, *args, **kwargs):
        #print(args)
        if args in cls.attrs:
            return cls.attrs[args]
        cls.attrs[args] = object.__new__(cls)
        return cls.attrs[args]

    def __init__(self, n, m):
        super(A, self).__init__(n)
        self.m = m

    def __call__(self, *args, **kwargs):
        k = Algoth(self.n) / Algoth(self.n - self.m)
        self.hist = k
        return k


class P(Algoth):
    hist: int = None

    def __init__(self, n):
        super(P, self).__init__(n)


class Pr(Algoth):
    attrs = {}
    hist: int = None
    args = []

    def __new__(cls, *args, **kwargs):
        #print(args)
        if args in cls.attrs:
            return cls.attrs[args]
        cls.attrs[args] = object.__new__(cls)
        return cls.attrs[args]

    def __init__(self, n, *args):
        super(Pr, self).__init__(n)
        self.args = [Algoth(x) for x in args]

    def __call__(self, *args, **kwargs):
        if self.hist is not None: return self.hist
        k = super(Pr, self).__call__()
        k /= reduce((lambda x, y: x * y), self.args)
        self.hist = k
        return k

    def __add__(self, other):
        return


class C(Algoth):
    attrs = {}
    hist: int = None
    args = []

    def __new__(cls, *args, **kwargs):
        # print(args)
        if args in cls.attrs:
            return cls.attrs[args]
        cls.attrs[args] = object.__new__(cls)
        return cls.attrs[args]

    def __init__(self, n, k):
        super(C, self).__init__(n)
        self.args = [Algoth(k)(), Algoth(n-k)()]

    def __call__(self, *args, **kwargs):
        if self.hist is not None: return self.hist
        k = super(C, self).__call__()
        k /= reduce((lambda x, y: x * y), self.args)
        self.hist = int(k)
        return int(k)

    @staticmethod
    def __doc__(self):
        filename = "C_n_k.png"
        with Image.open(filename) as img:
            return img.show()


class Bernoulli(C):
    attrs = {}
    hist: int = None
    args = []

    def __new__(cls, *args, **kwargs):
        # print(args)
        if args in cls.attrs:
            return cls.attrs[args]
        cls.attrs[args] = object.__new__(cls)
        return cls.attrs[args]

    def __init__(self, n, k, p):
        super(Bernoulli, self).__init__(n, k)
        self.k = k
        self.p = p

    def __call__(self, *args, **kwargs) -> float:
        if self.hist is not None: return self.hist
        f = super(Bernoulli, self).__call__()
        f *= pow(self.p, self.k)*pow(1-self.p, self.n - self.k)
        self.hist = f
        return f


class MuavrLaplace:
    def __init__(self, n: int | float | None = None, p: float | None = None, k: int | None | float = None):
        self.n = n
        self.p = p
        if p is not None:
            self.q = 1 - p
        self.k = k

    def integrate(self, k1: float | None = None, k2: float | None = None, x: float | None = None) -> float:
        if k2 is None:
            if x is None:
                x1 = -10
                x2 = (k1 - self.n * self.p)/pow(self.n * self.p * self.q, 1/2)
            else:
                x1 = -10
                x2 = x
        else:
            x1 = (k1 - self.n * self.p)/pow(self.n * self.p * self.q, 1/2)
            x2 = (k2 - self.n * self.p) / pow(self.n * self.p * self.q, 1 / 2)

        def F(x):
            return (1/pow(2*pi, 1/2)) * exp(-pow(x, 2)/2)

        return quad(F, x1, x2)[0]

    def plot(self):
        limits = (-10, 1000)
        acc = 4
        maxi = 0
        xm = 0
        X = [x/acc for x in range(limits[0]*acc, limits[1]*acc)]
        print(X)
        Y = []
        for x in range(limits[0]*acc, limits[1]*acc):
            y = self.integrate(x/acc, (x+1)/acc)
            if y > maxi:
                maxi = y
                xm = x/acc
            Y.append(y)
        a = plot(X, Y)
        axhline(y=maxi, color="b")
        axvline(x=xm, color="r")
        yticks(ticks=(maxi, ), labels=(str(round(maxi, 5)), ), minor=True, rotation=20)
        xticks(ticks=(xm, ), labels=(str(xm), ), minor=True, rotation=45)
        grid(True)
        show()

    def __repr__(self):
        return str(self.__call__())

    def __call__(self, *args, **kwargs):
        if self.k is None:
            print("here")
            self.plot()
            return ""
        return ((1/(pow(2 * pi * self.n * self.p * self.q, 1/2)))
                * exp(
                    -(pow(self.k - self.n * self.p, 2))
                    / (2 * self.n * self.p * self.q)))


def factorial(n):
    summ = 1
    for i in range(1, n+1):
        summ*=i
    return summ

if __name__ == "__main__":
    #print(Algoth(5) == factorial(5)) #True; Algoht = factorial лучше использовать factorial
    #print(P(5) == factorial(5)) #Перестановки без пвоторений
    #print(Pr(6, 3, 2) == factorial(6) / (factorial(3) * factorial(2))) #Перестановки с повторениями
    #print(A(4, 2) == factorial(4)/ factorial(2)) #Размещения без повторений
    #print(C(5, 3) == factorial(5)/(factorial(3)*factorial(5-3))) #Сочетания без повторений из n по k
    #print(Bernoulli(6, 3, 0.5) == C(6, 3) * pow(0.5, 3) * pow(1-0.5, 6-3))
    ##Бернулли 6 испытаний, 3 успеха, вероятность успеха 0.5
    #task = MuavrLaplace(10000, 0.006) #Инициализация n и p
    #print(task.integrate(80)) #Интеграл от x=0 до x=80
    ##при подстановке k=80 автоматически производится преобразование k = (80-np) / sqrt(npq)
    #print(1 - task.integrate(0, 81))  #интеграл от k1=0 до k2=81
    print((1/6)/(1-pow(5/6, 2)))