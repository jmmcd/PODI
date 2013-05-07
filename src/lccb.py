#!/usr/bin/env python

# LCCB: linear combination of coevolved bases

# Use coevolution to evolve a population of basis functions. Use
# linear regression with regularisation to combine them.

# This idea is inspired by the best bits of fast function extraction
# (McConaghy) -- a linear regression to get the best coefficients of
# basis functions -- and geometric semantic GP (Moraglio et al) --
# evolution to search among a more flexible choice of basis functions.

import os
import sys
import random
import collections
import copy

import zss
from zss.test_tree import Node
import numpy as np
from numpy import add, subtract, multiply, divide, sin, cos, exp, log, power, square, sqrt
np.seterr(all='raise', under='ignore')

import structure
import fitness
import variga
import bubble_down
from gp import srff

def LCCB_coevo(fitness_fn, pop):
    # get y = fitness_fn.train_y
    # Make a new array composed of pop[i].semantics for all i
    # (pop[i].semantics has already been calculated)
    X = np.array([ind.semantics for ind in pop]).T

    # copy the following code from FFX
    X /= X.std(0) # standardize -- makes it easier to choose l1_ratio
    eps = 5e-3
    
    # then linear regression with regularisation
    models = enet_path(X, y, eps=eps, l1_ratio=0.8)
    alphas_enet = np.array([model.alpha for model in models])
    coefs_enet = np.array([model.coef_ for model in models])

    # the overall model error is our overall result

    # extract the coefficients

    # undo the standardisation of X
    
    # assign the magnitude of coefficients as individual fitness
    # values

    # have to assign a new individual because tuples are immutable

    # then re-sort pop?
    pass
    
def run(fitness_fn, rep="bubble_down"):
    variga.MINLEN = 100
    variga.MAXLEN = 100
    variga.PHENOTYPE_DISTANCE = gp.tree_distance
    # run the fitness function as normal to get individuals' semantics
    variga.FITNESS = fitness_fn
    # but overwrite the individuals' fitness values
    variga.COEVOLUTIONARY_FITNESS = lambda pop: LCCB_coevo(fitness_fn, pop)
    if rep == "bubble_down":
        variga.GENERATE = generate_bubble_down_tree_and_fn
    elif rep == "grow":
        variga.GENERATE = generate_grow_tree_and_fn
    else:
        raise ValueError
    variga.MAXIMISE = False
    variga.SUCCESS = success
    variga.POPSIZE = 1000
    variga.GENERATIONS = 10
    variga.PMUT = 0.01
    variga.CROSSOVER_PROB = 0.7
    variga.ELITE = 1
    variga.TOURNAMENT_SIZE = 3
    variga.WRAPS = 1
    variga.main()





