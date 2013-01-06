#!/usr/bin/env python

import bubble_down

vars = list("x")
fns = {"j": 2}

class Lid:

    def __init__(self, target_depth, target_terminals=256, weight_depth=50, weight_terminals=50):
         self.target_depth = float(target_depth)
         self.target_terminals = target_terminals
         self.weight_depth = weight_depth
         self.weight_terminals = float(weight_terminals)
         total_weight = weight_depth + self.weight_terminals
         if total_weight != 100.0:
             raise ValueError('Lid weight_depth + weight_terminals != 100', total_weight)

    def __call__(self, ind):
        tree, nnodes, actual_depth = ind
        actual_terminals = 1 + nnodes / 2 # truncating division
        metric_depth = self.weight_depth * (
            1.0 - float(abs(self.target_depth - actual_depth))/self.target_depth)
        metric_terminals = 0
        if self.target_depth == actual_depth:
            metric_terminals = self.weight_terminals * (
                1.0 - float(abs(self.target_terminals - actual_terminals))/self.target_terminals)
        return metric_depth + metric_terminals

lid = Lid(10)
print(lid(bubble_down.bubble_down(25)))
