from functools import reduce
from scipy.integrate import quad
from math import pi, exp, sqrt
from matplotlib.pyplot import plot, show, grid, axvline, xticks, yticks, axhline
from typing import Callable
from PIL import Image
from numpy import array
import asyncio

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
        #print(args)
        if args in cls.attrs:
            return cls.attrs[args]
        cls.attrs[args] = object.__new__(cls)
        return cls.attrs[args]

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
        #print(f"{self.n}!={k}")
        return k


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
        k = int(super(A, self).__call__() / Algoth(self.n - self.m)())
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
        self.args = [Algoth(x)() for x in args]

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

    def __add__(self, other: int | Callable) -> int:
        if isinstance(other, Algoth):
            return self.__call__() + other.__call__()
        else:
            return self.__call__() + other

    def __mul__(self, other: int | Callable) -> int:
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
            return self.__call__() / other.__call__()
        else:
            if other == 0:
                raise ZeroDivisionError
            return self.__call__() / other

    def __rtruediv__(self, other):
        k = self.__call__()
        if k == 0:
            raise ZeroDivisionError
        if isinstance(other, Algoth):
            return other.__call__() / k
        else:
            return other / self.__call__()

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


if __name__ == "__main__":
    eps = 4
    m = 12
    n = 4
    lister = [[2, 4], [1, 2], [4, 1], [1, 5]]
    a1, a2 = 2, 4
    b1, b2 = 1, 2
    c1, c2 = 4, 1
    d1, d2 = 1, 5

    def hyper_geometric(n1, n2, k1, k2, n = None, k = None, numb = None):
        """C из n1 о k1 * C из n2 по k2"""
        if k is None:
            k = k1 + k2
        if n is None:
            n = n1 + n2
        if numb is None:
            numb = 1
        return round(C(n1, k1) * C(n2, k2) / C(n, k) / numb, 6)

    print("EPS i NU")
    vel = []
    for i in range(0, 5):
        vel.append(round(hyper_geometric(4, 8, i, 5-i), 3))
        print(vel[i])
    def matrix_print(matrix: list[list]):
        for i in matrix:

            print(" | ".join([str(x) for x in i]))



    print("EPS\\Nu")
    table = []
    summ = 0
    for i in range(0, 6):
        row = []
        numb = hyper_geometric(4, 8, i, 5-i)
        for j in range(0, 6):
            if i + j < 6:
                sumc = round(Pr(5, i, j, 5-i-j)()*pow((1/3), i)*pow((1/3), j)*pow((1/3), 5-i-j), 3)
            else:
                sumc = 0
            summ += sumc
            row.append(sumc)
        table.append(row)
    matrix_print(table)

    summ = 0
    table2 = []
    for i in range(0, 6):
        row = []
        for j in range(0, 6):
            if i+j>0 and i+j<6:
                sumc = round(C(4, i)*C(4, j)*C(4, 5-i-j)/C(12, 5), 3)
            else:
                sumc=0
            summ += sumc
            row.append(sumc)
        table2.append(row)
    #matrix_print(table2)
    print("F(x, y)")
    tablef1 = [[0 for i in range(6)] for j in range(6)]
    for i in range(6):
        for j in range(6):
            for k in range(0, i):
                for n in range(0, j):
                    tablef1[i][j] += table[k][n]
    print(tablef1[2][4], tablef1[1][2], tablef1[4][1], tablef1[1][5])
    tablef2 = [[0 for i in range(6)] for j in range(6)]
    for i in range(6):
        for j in range(6):
            for k in range(0, i):
                for n in range(0, j):
                    tablef2[i][j] += table2[k][n]
    print(tablef2[2][4], tablef2[1][2], tablef2[4][1], tablef2[1][5],"\n","\n", "Mat")


    def mat(spis: list | dict, chisla: list | None = None):
        if chisla is None:
            chisla = [i for i in range(len(spis))]
        return reduce(lambda x, y: x + y, [spis[i]*chisla[i] for i in range(len(chisla))])

    print(round(mat(vel), 3))

    def disp(spis: list | dict, mat: int | None = None, chisla: list | None = None, fun: Callable | None = None):
        if mat is None:
            mat = fun(spis, chisla)
        if chisla is None:
            chisla = [i for i in range(len(spis))]
        return reduce(lambda x, y: x + y, [pow(chisla[j]-mat, 2)*spis[j] for j in range(len(spis))])

    def mat_table(table: list[list], chisla: list[list] | None = None):
        if chisla is None:
            chisla = [[(i, j) for i in range(len(table))] for j in range(len(table))]
        summ = 0
        for i in range(len(table)):
            for j in range(len(table)):
                summ += chisla[i][j][0]*chisla[i][j][1]*table[i][j]
        return summ



    print("Disp", round(disp(vel, fun=mat), 3), disp(vel, mat(vel)))
    print("M(eps, nu)", round(mat_table(table2), 3))
    print("Cov=", round(mat_table(table2) - mat(vel)*mat(vel), 3))
    print("ro=", round((mat_table(table2) - mat(vel)*mat(vel))/sqrt(disp(vel, fun=mat)*disp(vel, fun=mat)), 3))




