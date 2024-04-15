import prob_theory
from prob_theory import C, Pr, MuavrLaplace
from PIL import Image
from math import sqrt
from functools import reduce
from typing import Callable


class FiveTask:
    """задача с распределением Муавра-Лапласа"""
    def __init__(self, p, prob: None | float = None, n: None | int = None, eps: None | float = None):
        """n - количество испытаний, чтобы с в-тью prob отклонение от p не превышало eps,
        где n/eps - неизвестно (просто не вводите число)"""
        self.n = n
        self.p = p
        self.prob = prob
        self.eps = eps
        if eps is None:
            self.type = "2"
        elif n is None:
            self.type = "1"
        else:
            self.type = "3"

    def __solve_1(self):
        for i in range(0, 100000):
            j = i/3
            x = self.eps * sqrt(j) / sqrt(self.p * (1 - self.p))
            if (n:=MuavrLaplace().integrate(x=x)) > (self.prob + 1)/2:
                print(f"в-ть = {n}", f"проведено {j} испытаний")
                break

    def __solve_2(self):
        for i in range(0, 10000):
            j = i/10000
            x = j * sqrt(self.n) / sqrt(self.p * (1 - self.p))
            if n:=MuavrLaplace().integrate(x=x) > (self.prob + 1)/2:
                print(f"в-ть = {n}", f"eps ~ {j} ")
                break

    def __solve_3(self):
        task = MuavrLaplace(self.n, self.p)

        for i in range(1, self.n * 4):
            j = i/4
            if k:=task.integrate(j) >= self.eps:
                print(f"При n={j}, в-ть больше {self.eps}: ={k}")
                break

    def solve(self):
        if self.type == "1":
            return self.__solve_1()
        elif self.type == "2":
            return self.__solve_2()
        else:
            return self.__solve_3()

    def __doc__(self):
        filename = "5task.png"
        with Image.open(filename) as img:
            return img.show()


class SevenTask:
    """седьмая задача про совместное распределение случайных величин"""
    dots: list[tuple] = None
    table: list[list] = None
    table2: list[list] = None
    tablef1: list[list] = None
    tablef2: list[list] = None
    vel1: list = None
    vel2: list = None

    def __init__(self, n1, n2, n3, m, *args):
        self._n1_ = n1
        self._n2_ = n2
        self._n3_ = n3
        self._n_ = n1 + n2 + n3
        self._m_ = m
        self.dots = []
        self.table = []
        self.table2 = []
        self.tablef1 = []
        self.tablef2 = []
        self.vel1 = []
        self.vel2 = []
        for i in args:
            self.dots.append(i)

    def initialize_dotes(self, a1, a2):
        self.dots.append((a1, a2, ))

    def get_data(self):
        return (self._n1_, self._n2_, self._n3_, self._n_, self._m_)

    @staticmethod
    def _hyper_geometric_(n1, n2, k1, k2, n = None, k = None, numb = None):
        """C из n1 о k1 * C из n2 по k2"""
        if k is None:
            k = k1 + k2
        if n is None:
            n = n1 + n2
        if numb is None:
            numb = 1
        return round(C(n1, k1) * C(n2, k2) / C(n, k) / numb, 6)

    @staticmethod
    def _matrix_print_(matrix: list[list]):
        for i in matrix:

            print(" | ".join([str(x) for x in i]))

    def p_eps(self):
        print("EPS-raspredelenie i NU-raspredelenie")
        print("\neps")
        self.vel1 = []
        for i in range(0, min(self._n1_, self._m_) + 1):
            self.vel1.append(round(self._hyper_geometric_(self._n1_, self._n_-self._n1_, i, self._m_ - i), 3))
            print(self.vel1[i], end=' ')
        print("\nnu")
        self.vel2 = []
        for i in range(0, min(self._n2_, self._m_) + 1):
            self.vel2.append(round(self._hyper_geometric_(self._n2_, self._n_ - self._n2_, i, self._m_ - i), 3))
            print(self.vel2[i], end=' ')

    def eps_nu(self):
        self.table = []
        print("\neps\\nu -raspredelenie s vozvratom\n")
        summ = 0
        for i in range(0, min(self._n1_, self._m_) + 1):
            row = []
            for j in range(0, min(self._n2_, self._m_) + 1):
                if i + j < self._m_+1:
                    #print(f"{i}, {j}, PR={Pr(self._m_, i, j, self._m_ - i - j)()}")
                    sumc = round(
                        Pr(self._m_, i, j, self._m_ - i - j)() *
                        pow(self._n1_ / self._n_, i) *
                        pow(self._n2_ / self._n_, j) *
                        pow(self._n3_ / self._n_, self._m_ - i - j), 3)
                else:
                    sumc = 0
                summ += sumc
                row.append(sumc)
            self.table.append(row)

        self._matrix_print_(self.table)

        print("\nbez vozvrata\n")

        self.table2 = []
        summ = 0
        for i in range(min(self._n1_, self._m_) + 1):
            row = []
            for j in range(0, min(self._n2_, self._m_) + 1):
                if i + j >= self._m_ - self._n3_ and i + j <= self._m_:
                    sumc = round(C(self._n1_, i) * C(self._n2_, j) * C(self._n3_, self._m_ - i - j) / C(self._n_, self._m_), 3)
                else:
                    sumc = 0
                summ += sumc
                row.append(sumc)
            self.table2.append(row)

        self._matrix_print_(self.table2)

    def function(self):
        self.tablef1 = [[0 for i in range(min(self._n2_, self._m_) + 1)] for j in range(min(self._n1_, self._m_) + 1)]
        #self._matrix_print_(self.tablef1)
        for i in range(min(self._n1_, self._m_) + 1):
            for j in range(min(self._n2_, self._m_)+1):
                #print(i, j)
                for k in range(0, i):
                    for n in range(0, j):
                        self.tablef1[i][j] += self.table[k][n]


        print("Frunction s vozvrasheniem")
        for i in self.dots:
            print(self.tablef1[i[0]][i[1]])

        self.tablef2 = [[0 for i in range(min(self._n2_, self._m_) + 1)] for j in range(min(self._n1_, self._m_)+1)]
        for i in range(min(self._n1_, self._m_) + 1):
            for j in range(min(self._n2_, self._m_)+1):
                for k in range(0, i):
                    for n in range(0, j):
                        self.tablef2[i][j] += self.table2[k][n]

        print("Function bez vozvrasheniya")
        for i in self.dots:
            print(self.tablef2[i[0]][i[1]])

    @staticmethod
    def mat(spis: list | dict, chisla: list | None = None):
        if chisla is None:
            chisla = [i for i in range(len(spis))]
        return reduce(lambda x, y: x + y, [spis[i]*chisla[i] for i in range(len(chisla))])

    @staticmethod
    def disp(spis: list | dict, mat: int | None = None, chisla: list | None = None, fun: Callable | None = None):
        if mat is None:
            mat = fun(spis, chisla)
        if chisla is None:
            chisla = [i for i in range(len(spis))]
        return reduce(lambda x, y: x + y, [pow(chisla[j] - mat, 2) * spis[j] for j in range(len(spis))])

    @staticmethod
    def mat_table(table: list[list], chisla: list[list] | None = None):
        if chisla is None:
            chisla = [[(i, j) for i in range(len(table))] for j in range(len(table))]
        summ = 0
        for i in range(len(table)):
            for j in range(len(table)):
                summ += chisla[i][j][0]*chisla[i][j][1]*table[i][j]
        return summ

    def characteristics(self):
        print("Meps", round(self.mat(self.vel1), 3))
        print("Mnu", round(self.mat(self.vel2), 3))
        print("DispEps", round(self.disp(self.vel1, fun=self.mat), 3))
        print("DispNu", round(self.disp(self.vel2, fun=self.mat), 3))
        print("M(eps, nu)", round(self.mat_table(self.table2), 3))
        #print("Cov=", round(self.mat_table(table2) - mat(vel) * mat(vel), 3))
        print("ro=",
              round((self.mat_table(self.table2) -
                     self.mat(self.vel1) * self.mat(self.vel2))
                    /
                    sqrt(self.disp(self.vel1, fun=self.mat) *
                         self.disp(self.vel2, fun=self.mat)), 3))

    def __str__(self):
        self.p_eps()
        self.eps_nu()
        self.function()
        self.characteristics()
        return ""


class EightTask:
    """восьмая задача про теорему Байеса"""
    a: tuple = None

    def __init__(self, n, l, a: tuple):
        self.n = n
        self.l = l
        self.a = a
        self.p_a = 0

    @staticmethod
    def polinomial(n: int, *args) -> float:
        length: int = len(args)//2
        k: list[int] = list(args[0:length])
        p: list[float] = list(args[length:len(args)])
        if n != sum(k):
            k.append(n-sum(k))
            p.append(1-sum(p))
        return Pr(n, *k)()*reduce(lambda x, y: x*y, [pow(p[i], k[i]) for i in range(len(p))])

    def solve_1(self):
        print("with returning")
        self.p_a = 0
        for k1 in range(0, self.n + 1):
            for k2 in range(0, self.n - k1 + 1):
                if k1+k2 == self.n:
                    p_h_i = self.polinomial(self.n, k1, k2, 1/2, 1/2)
                    p_a_h_i = self.polinomial(self.l, self.a[0], self.a[1], k1/self.n, k2/self.n)
                    self.p_a += p_h_i * p_a_h_i
        maxi = 0
        most_prob = ()
        for k1 in range(0, self.n + 1):
            for k2 in range(0, self.n - k1 + 1):
                if k1+k2 == self.n:
                    p_h_i = self.polinomial(self.n, k1, k2, 1/2, 1/2)
                    p_a_h_i = self.polinomial(self.l, self.a[0], self.a[1], k1/self.n, k2/self.n)
                    if p_h_i*p_a_h_i/self.p_a > maxi:
                        maxi = p_h_i*p_a_h_i/self.p_a
                        most_prob = (k1, k2, )
        print(round(maxi, 3), most_prob)
        print("without  returning")

        self.p_a = 0
        for k1 in range(0, self.n + 1):
            for k2 in range(0, self.n - k1 + 1):
                if k1+k2 == self.n:
                    p_h_i = self.polinomial(self.n, k1, k2, 1/2, 1/2)
                    p_a_h_i = C(k1, self.a[0])*C(k2, self.a[1])/C(self.n, self.l)
                    self.p_a += p_h_i * p_a_h_i
        maxi = 0
        most_prob = ()
        for k1 in range(0, self.n + 1):
            for k2 in range(0, self.n - k1 + 1):
                if k1+k2 == self.n:
                    p_h_i = self.polinomial(self.n, k1, k2, 1/2, 1/2)
                    p_a_h_i = C(k1, self.a[0])*C(k2, self.a[1])/C(self.n, self.l)
                    if p_h_i*p_a_h_i/self.p_a > maxi:
                        maxi = p_h_i*p_a_h_i/self.p_a
                        most_prob = (k1, k2, )
        print(round(maxi, 3), most_prob)


if __name__ == "__main__":
    my_five_task = FiveTask(n=500, p=0.3, eps=0.95)
    my_five_task.solve()
    #my_seven_task = SevenTask(3, 7, 4, 6, (2, 2), (1, 3), (1, 4), (3, 5))
    #print(my_seven_task)
    #my_eight_task = EightTask(12, 4, (3, 1, ))
    #my_eight_task.solve_1()
    pass

