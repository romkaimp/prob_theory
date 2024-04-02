import tasks
import prob_theory


#liza = tasks.FiveTask(p=0.006, prob=0.0228, n=10000).solve()
#print(1200)
amina = tasks.SevenTask(3, 6, 4, 6, (2, 2), (1, 3), (1, 4), (2, 5))

amina.p_eps()
print(amina)