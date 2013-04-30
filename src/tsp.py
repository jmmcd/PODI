#!/usr/bin/env python

import variga
import numpy as np
from math import sqrt
import gzip
import random

def overlapping_pairs(seq):
    return [seq[i:i+2] for i in range(len(seq) - 1)]

def euclidean_distance(x, y):
    return sqrt(sum((xi - yi) ** 2.0 for xi, yi in zip(x, y)))

class TSP:
    def __init__(self, filename):
        self.coords = {}
        self.read_file(filename)
        self.read_optimal_results("../data/TSPLIB/STSP.html")

    def read_optimal_results(self, filename):
        import re
        optimal_results = {}
        for line in open(filename).readlines():
            p = r">(\w+) : (\d+)<"
            m = re.search(p, line)
            if m:
                key, val = m.group(1, 2)
                key = key.strip()
                # optimal results are given as integers in TSPLIB
                val = int(val.split()[0].strip()) 
                optimal_results[key] = val
        print("Optimal results:")
        print(optimal_results)
        self.optimal = optimal_results[self.name]

    def read_file(self, filename):
        """FIXME this only works for files in the node xy-coordinate
        format. Some files eg bayg29.tsp.gz give explicit edge weights
        instead."""
        f = gzip.open(filename, "rb")
        coord_section = False
        for line in f.readlines():
            if line.startswith("NAME"):
                self.name = line.split(":")[1].strip()
            elif (line.startswith("COMMENT") or
                  line.startswith("TYPE") or
                  line.startswith("EDGE_WEIGHT_TYPE") or
                  line.startswith("EOF")):
                pass                  
            elif line.startswith("DIMENSION"):
                self.n = int(line.split(":")[1].strip())
            elif line.startswith("NODE_COORD_SECTION"):
                coord_section = True
            elif coord_section:
                # coords are sometimes given as floats in TSPLIB
                idx, x, y = map(float, line.split(" "))
                self.coords[idx] = x, y

    def perm(self, random):
        s = range(1, self.n+1)
        random.shuffle(s)
        return s

    def tour_length(self, tour):
        return sum(self.dist(*pair)
                   for pair in overlapping_pairs(tour) + [[tour[-1], tour[0]]])

    def success(self, fitness):
        return self.optimal >= fitness

    def dist(self, x, y):
        return euclidean_distance(self.coords[x], self.coords[y])

if __name__ == "__main__":
    problem = TSP("../data/TSPLIB/att48.tsp.gz")
    # tour = problem.perm(random)
    # print(tour)
    # print(problem.tour_length(tour))
    variga.GENERATE = problem.perm
    variga.FITNESS = problem.tour_length
    variga.SUCCESS = problem.success
    variga.MAXIMISE = False
    variga.MINLEN = 200
    variga.MAXLEN = 500
    variga.main()
