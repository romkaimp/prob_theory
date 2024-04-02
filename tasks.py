import prob_theory
from math import sqrt


class FiveTask:
    def __init__(self, p, prob, n: None | float = None, eps: None | float = None):
        """n - количество испытаний, чтобы с в-тью prob отклонение от p не превышало eps,
        где n/eps - неизвестно (просто не вводите число)"""
        self.n = n
        self.p = p
        self.prob = prob
        self.eps = eps
        if eps is None:
            self.type = "2"
        else:
            self.type = "1"

    def __solve_1__(self):
        for i in range(0, 100000):
            j = i/3
            x = self.eps * sqrt(j) / sqrt(self.p * (1 - self.p))
            if n:=prob_theory.MuavrLaplace().integrate(x=x) > (self.prob + 1)/2:
                print(f"в-ть = {n}", f"проведено {j} испытаний")
                break

    def __solve_2__(self):
        for i in range(0, 10000):
            j = i/10000
            x = j * sqrt(self.n) / sqrt(self.p * (1 - self.p))
            if n:=prob_theory.MuavrLaplace().integrate(x=x) > (self.prob + 1)/2:
                print(f"в-ть = {n}", f"eps ~ {j} ")
                break

    def solve(self):
        if self.type == "1":
            return self.__solve_1__()
        else:
            return self.__solve_2__()


class SevenTask:


if __name__ == "__main__":
    my_task = FiveTask(0.85, 0.997, eps=0.01)
    my_task.solve()