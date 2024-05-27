import prob_theory
from math import pi
from scipy import integrate
import numpy
from sympy import symbols, simplify, expand, diff, limit, Rational, exp, I, integrate as integr, solve
from fractions import Fraction
from typing import Tuple, Optional, Union, List
import enum


class SV(enum.Enum):
    nu: str = "nu"
    ksi: str = "ksi"


class NormalDistribution:
    '''m - матожидание, d - среднеквадратичное отклонение'''
    def __init__(self, m: Optional[float] = None, d: Optional[float] = None):
        self.m = m
        self.d = d

    def initialize(self, m, d):
        self.m = m
        self.d = d

    @property
    def probf(self):
        x = symbols(self.symbol)
        return (1/pow(2*pi, 1/2)/self.d) * exp(-pow((x-self.m)/self.d, 2)/2)

    @probf.setter
    def probf(self, symbol: str):
        self.symbol = symbol

    def integrate(self, a, b: Optional[int | float] = None, me: bool = False):
        def prob(x):
            if me:
                return x * (1/pow(2*pi, 1/2)/self.d) * exp(-pow((x-self.m)/self.d, 2)/2)
            return (1/pow(2*pi, 1/2)/self.d) * exp(-pow((x-self.m)/self.d, 2)/2)
        if b is None:
            return prob(a)
        return integrate.quad(prob, a, b)

    @staticmethod
    def solve(a, E, post_sv: SV, post_sv_val):
        def m_n_pri_e(a, E, post_sv: SV, post_sv_val):
            '''post_sv:  M(nu | post_sv = post_sv_val); post_sv = {SV.nu, SV.ksi}'''
            if post_sv == SV.ksi:
                return a[0] - E[0][1] / E[1][1] * (post_sv_val - a[1])
            else:
                return a[1] - E[0][1] / E[0][0] * (post_sv_val - a[0])

        def d_n_pri_e(E, post_sv: SV, post_sv_val=0) -> Tuple[Fraction, float]:
            '''post_sv:  D(nu | post_sv = post_sv_val); post_sv = {SV.nu, SV.ksi}'''
            if post_sv == SV.ksi:
                return Fraction(E[1][1]) - Fraction(E[0][1] * E[0][1]) / Fraction(E[0][0]), E[1][1] - E[0][1] * E[0][
                    1] / E[0][0]
            else:
                return Fraction(E[0][0]) - Fraction(E[0][1] * E[0][1]) / Fraction(E[1][1]), E[0][0] - E[0][1] * E[0][
                    1] / E[1][1]

        new_dstr = NormalDistribution(m_n_pri_e(a, E, post_sv, post_sv_val), pow(d_n_pri_e(E, post_sv)[0], 1/2))
        return new_dstr


class PokazatDistribution:
    def __init__(self, lmb):
        self.lmb = lmb

    def integrate(self, a, b: Optional[int] = None):
        def prob(x):
            return (self.lmb*exp(-self.lmb*x))
        a = max(a, 0, b)
        if b is None:
            return prob(a)
        return integrate.quad(prob, a, b)

    @property
    def probf(self):
        x = symbols(self.symbol)
        return self.lmb*exp(-self.lmb*x)

    @probf.setter
    def probf(self, symbol):
        self.symbol = symbol


class GeneratingFunction:

    def __init__(self, prob_f: Union[NormalDistribution, PokazatDistribution, List[float | Rational]]):
        if isinstance(prob_f, List):
            z, phi = symbols("z phi")
            cur = 1
            for i in range(len(prob_f)):
                cur *= z*prob_f[i] + 1 - prob_f[i]
            phi = cur
            self.phi = expand(phi)
            print(expand(phi))

    @property
    def me(self):
        z, phi = symbols("z phi")
        return limit(diff(self.phi, z), z, 1)

    @property
    def disp(self):
        z, phi = symbols("z phi")
        k = diff(self.phi, z)
        return limit(expand(diff(k, z) + k - k**2), z, 1)


class CharacteristicFunction:
    def __init__(self, prob_f: Union[NormalDistribution, PokazatDistribution, List[float]]):
        if isinstance(prob_f, List):
            t, x = symbols("t x")
            cur = 0
            for i in range(len(prob_f)):
                cur += exp(t * i * I) * prob_f[i]

            phi = cur
            self.phi = expand(phi)
            print(expand(phi))

    @property
    def me(self):
        t, x = symbols("t x")
        k = diff(self.phi, t)
        return -I*limit(k, t, 0)

    @property

    def disp(self):
        t, x = symbols("t x")
        k = diff(self.phi, t)
        return limit(-diff(k, t) + expand(k**2), t, 0)



class distrs(enum.Enum):
    norm: str = "norm"
    pokaz: str = "pokaz"


class FirstTaskNV:
    '''returns Me(n | ksi = x), D(n | ksi = x),
     probability: y in (a, b), where post_sv = SV.ksi, post_sv_val=x, horizon=(a, b)'''
    def __init__(self, a: List[int], E: List[List[float]], post_sv: SV, post_sv_val: float, horizon: Tuple[float, float], type: distrs = distrs.norm):
        if type == distrs.norm:
            distr = NormalDistribution.solve(a, E, post_sv, post_sv_val)
            square = distr.integrate(horizon[0], horizon[1])
            self.m = distr.m
            self.d = distr.d * distr.d
            self.square = square

    def __str__(self):
        args = tuple(map(round, (self.m, self.d, self.square[0], self.square[1]), (3, 3, 3, 3)))
        #print(args)
        return "Me: {}, Disp: {}, prob: {} err={}".format(*args)


class FirstTask:
    def __init__(self):
        def f(x):
            return x * exp(-4 * x**2)
        a = round(1/integrate.quad(f, 0, numpy.inf)[0], 3)
        self.a = a
        x = symbols("x")
        self.probf = a * x * exp(-4 * x**2)

    @property
    def me(self):
        x = symbols("x")
        return integr(x*self.probf, (x, 0, numpy.inf))

    @property
    def moda(self):
        x = symbols("x")
        return solve(diff(self.probf, x), x)[1]

    def integrate(self, a, b):
        x = symbols("x")
        return integr(self.probf, (x, a, b))

    def __str__(self):
        return "param a: {0}, me: {2}, moda: {1}, integral 0-{1}: {3}".format(*tuple(map(round, (self.a, self.moda, self.me, self.integrate(0, self.moda)), (3, 3, 3, 3))))


class ThirdTask:
    def __init__(self, type: distrs):
        if type == distrs.pokaz:
            x, y, z = symbols("x y z")

            r1 = PokazatDistribution(1)
            r2 = PokazatDistribution(1)
            r1.probf = 'x'
            r2.probf = 'y'

            first = simplify(integr(r1.probf * r2.probf, (y, x/z, numpy.inf)))
            #print(first)
            second = simplify(integr(first, (x, 0, numpy.inf)))
            ProbFun = second.args[0][0]
            #print(second.args[0][0])
            k = simplify(diff(ProbFun, z))
            #print(simplify(diff(ProbFun, z)))
            self.probf = k
        elif type == distrs.norm:
            x, y, z = symbols("x y z")

            r1 = NormalDistribution(0, 1)
            r2 = NormalDistribution(0, 1)
            r1.probf = 'x'
            r2.probf = 'y'

            first = simplify(integr(r1.probf * r2.probf, (y, x / z, numpy.inf)))
            #print(first)
            second = simplify(integr(first, (x, 0, numpy.inf)))
            ProbFun = second.args[0][0]
            #print(second.args[0][0])
            k = simplify(diff(ProbFun, z))
            #print(simplify(diff(ProbFun, z)))
            self.probf = k

    def __str__(self):
        return str(self.probf)




if __name__ == "__main__":
    #a = [2, -3]
    #E = [[3, -2], [-2, 2]]
    #sv = SV.ksi
    #sv_val = -4
    #print(FirstTaskNV(a, E, sv, sv_val, (-1, 2)))

    #g = GeneratingFunction([Rational(1, 10), Rational(2, 10), Rational(3, 10), Rational(4, 10)])
    #print(g.me(), g.disp())

    #f = CharacteristicFunction([Rational(1, 6), Rational(1,3), Rational(1,6), Rational(1, 3)])
    #print(f.me(), f.disp())

    #ft = FirstTask()
    #print(ft)

    #print(ThirdTask(distrs.norm))
    pass