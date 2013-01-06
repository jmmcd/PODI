#!/usr/bin/env python

import random
from math import *
from operator import mul, add, sub
import nonrandom
import sys
import variga

x = [0.1 * i for i in range(10)]
f = lambda x: 3.0 * sin(x * x) - x
y = map(f, x)

symbols = {"sin": (sin, 1),
           "*": (mul, 2),
           "+": (add, 2),
           "-": (sub, 2),
           "0.1": (0.1, 0),
           "0.2": (0.2, 0),
           "0.5": (0.5, 0),
           "1.0": (1.0, 0),
           "x": ("x", 0)
           }
variables = ["x"]
constants = [k for k in symbols.keys() if symbols[k][1] == 0 and k not in variables]
functions = [k for k in symbols.keys() if symbols[k][1] > 0]

def evaluate(x, random):
    L = random.randint(5, 20)
    stack = []
    for i in range(L):
        if len(stack) > 6: choices = functions
        elif len(stack) > 2: choices = constants + variables + functions
        else: choices = constants + variables
        symbol = random.choice(choices)
        # print(i, symbol, stack)
        if symbol == "x":
            stack.append(x)
        else:
            f, arity = symbols[symbol]
            if arity: 
                stack[-arity:] = [f(*stack[-arity:])]
            else:
                stack.append(f)
    return stack[-1]

# genome = [random.randint(0, sys.maxint-1) for i in range(200)]
# nr = nonrandom.NonRandom(genome)
# print(fitness(nr))

variga.GENERATE = lambda rng: [evaluate(xi, random) for xi in x]
variga.FITNESS = lambda fx: sum((fxi - yi) ** 2.0 for fxi, yi in zip(fx, y))
variga.SUCCESS = lambda x: x < 0.01
variga.MAXIMISE = False
variga.main()

