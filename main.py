import tasks
import prob_theory


p1 = 0.85
p2 = 0.9
p3 = 0.85
p4 = 0.9
p5 = 0.85
p12 = p1*p2
p123 = p12 + p3 - p12*p3
p1234 = p123 + p4 - p123 * p4
p12345 = p1234 * p5
print(round(p123, 3), round(p1234, 3), round(p12345, 3))






