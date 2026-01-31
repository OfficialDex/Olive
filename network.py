# Main olive creation file

# imports
import random
import math


# functions
def r():
    return random.uniform(-1, 1)


def t(x):
    return math.tanh(x)


class network:
    def __init__(self, i, h, o):
        self.w1 = [[r() for _ in range(i)] for _ in range(h)]
        self.w2 = [[r() for _ in range(h)] for _ in range(o)]

    def forward(self, x):
        h = [t(sum(a * b for a, b in zip(x, w))) for w in self.w1]
        return [t(sum(a * b for a, b in zip(h, w))) for w in self.w2]

    def copy(self):
        n = network(1, 1, 1)
        n.w1 = [r[:] for r in self.w1]
        n.w2 = [r[:] for r in self.w2]
        return n

    def mutate(self, m):
        for i in range(len(self.w1)):
            for j in range(len(self.w1[i])):
                if random.random() < m:
                    self.w1[i][j] += r() * 0.2
        for i in range(len(self.w2)):
            for j in range(len(self.w2[i])):
                if random.random() < m:
                    self.w2[i][j] += r() * 0.2
