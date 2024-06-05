from sklearn.cluster import KMeans
from scipy.linalg import solve
from sympy import Rational
import numpy
from typing import Tuple


class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def coordinates(self):
        return [self.x, self.y]

    def __str__(self):
        return str(self.coordinates)


class Line:
    """y = kx + b"""
    def __init__(self, k, b):
        self.k = k
        self.b = b

    def collision(self, other):
        if isinstance(other, Line):
            if other.k != self.k:
                A = [[1, -self.k],
                     [1, -other.k]]
                b = [self.b, other.b]
                coll = solve(A, b)
                new_node = Node(coll[0], coll[1])
                return new_node
            else:
                return None

    def __repr__(self):
        return "y = {}*x + {}".format(self.k, self.b)


class Graph:
    def __init__(self):
        self.nodes = []
        self.line_matrix = [[]]

    def add_node(self, node: Node):
        self.nodes.append(node)

    def add_nodes(self, nodes: Tuple[Node, ...]):
        for node in nodes:
            self.nodes.append(node)

    def line_matrix_repr(self):
        for node_1 in range(len(self.nodes)):
            print()
            for node_2 in range(node_1 + 1, len(self.nodes)):
                print(self.line_matrix[node_1][node_2], end="  | ")
            print("\n")

    def square(self):
        self.line_matrix = [[0 for i in range(len(self.nodes))] for j in range(len(self.nodes))]
        for node_1 in range(len(self.nodes)):
            for node_2 in range(node_1+1, len(self.nodes)):
                try:
                    k = Rational(self.nodes[node_2].y - self.nodes[node_1].y) / Rational(self.nodes[node_2].x - self.nodes[node_1].x)
                except ZeroDivisionError as err:
                    k = numpy.inf
                b = self.nodes[node_1].y - k * self.nodes[node_1].x
                self.line_matrix[node_1][node_2] = Line(k, b)




if __name__ == "__main__":
    chic = Graph()
    print()
    chic.add_nodes((Node(5, 1), Node(10, 3), Node(17, 5), Node(1, 2), Node(3, 4)))
    chic.square()
    chic.line_matrix_repr()
