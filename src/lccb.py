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
from sklearn.linear_model import enet_path

import structure
import fitness
import variga
from bubble_down import generate_bubble_down_tree_and_fn, generate_grow_tree_and_fn, generate_bubble_down_tree_and_fn_minn_maxn, generate_grow_tree_and_fn_maxd
from gp import srff, tree_distance

def LCCB_coevo(fitness_fn, pop):
    y = fitness_fn.train_y
    # Make a new array composed of pop[i].semantics for all i
    # (pop[i].semantics has already been calculated)
    X_cols = []
    for ind in pop:
        if (ind.phenotype and ind.fitness != sys.maxint
            and all(np.isfinite(ind.semantics))):
            X_cols.append(ind.semantics)
        else:
            print("omitting a column")
            X_cols.append(np.zeros(len(y)))
    X = np.asarray(X_cols).T
    eps = 5e-3

    # FIXME unbias the data as in FFX, and later rebias?
    
    # then linear regression with regularisation
    models = enet_path(X, y, eps=eps, l1_ratio=0.8)
    alphas = np.array([model.alpha for model in models])
    coefss = np.array([model.coef_ for model in models])

    # somehow choose just one model -- FIXME for now just choosing
    # model number 10
    model, alpha, coefs = models[10], alphas[10], coefss[10]

    # the model score is our overall result. it is an R^2 values,
    # hence 0 is bad, 1 is good.
    print("score", model.score(X, y))
    
    # Assign the magnitude of coefficients as individual fitness
    # values. Have to construct a new individual because tuples are
    # immutable
    pop = [variga.Individual(genome=pop[i].genome,
                             used_codons=pop[i].used_codons,
                             fitness=-abs(coefs[i]),
                             phenotype=pop[i].phenotype,
                             readable_phenotype=pop[i].readable_phenotype,
                             semantics=pop[i].semantics)
           for i in range(len(pop))]

    pop.sort(key=variga.ind_compare)

    
def run(fitness_fn, rep="bubble_down"):
    variga.MINLEN = 100
    variga.MAXLEN = 100
    variga.PHENOTYPE_DISTANCE = tree_distance
    # run the fitness function as normal to get individuals' semantics
    variga.FITNESS = fitness_fn
    # but overwrite the individuals' fitness values
    variga.COEVOLUTIONARY_FITNESS = lambda pop: LCCB_coevo(fitness_fn, pop)
    if rep == "bubble_down":
        variga.GENERATE = lambda rng: generate_bubble_down_tree_and_fn_minn_maxn(10, 20, rng)
    elif rep == "grow":
        variga.GENERATE = generate_grow_tree_and_fn_maxd
    else:
        raise ValueError
    variga.MAXIMISE = False    
    variga.SUCCESS = lambda x: False # FIXME
    variga.POPSIZE = 200
    variga.GENERATIONS = 50
    variga.PMUT = 0.01
    variga.CROSSOVER_PROB = 0.7
    variga.ELITE = 1
    variga.TOURNAMENT_SIZE = 3
    variga.WRAPS = 1
    variga.main()


if __name__ == "__main__":
    run(srff, rep="grow")
