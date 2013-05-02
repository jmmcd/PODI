#!/usr/bin/env python

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

from bubble_down import mknd, mkst, depth, get_node, get_subtree, traverse, make_fn, evaluate, grow, srff

from bubble_down import vars, fns

mutation_prob = 0.01

def mutate(t, p=mutation_prob):
    """Point mutation of a tree. Traverse the tree. For each node,
    with low probability, replace it with another node of same
    arity."""
    # FIXME this deepcopy is probably necessary, but will probably be
    # slow
    t = copy.deepcopy(t)

    # depth-first traversal of tree
    for nd, st, path in traverse(t):

        if random.random() < p:
            # mutate
            if fns.has_key(nd):
                # this node is a non-terminal
                arity = fns[nd]
                options = [fn for fn in fns if fns[fn] == arity and fn != nd]
                if options:
                    # st[0] is the node itself
                    st[0] = random.choice(options)
            else:
                # this node is a terminal
                arity = 0
                options = [term for term in vars if term != nd]
                if options:
                    # s will be the subtree rooted at parent
                    s = get_subtree(t, path[:-1])
                    # s[path[-1]] tells which child it is
                    s[path[-1]] = random.choice(options)
    return t
            
            
def semantic_geometric_mutate(t, ms=0.01):
    """Semantic geometric mutation as defined by Moraglio et al:

    tm = t + ms * (tr1 - tr2)
    
    where ms is the mutation step (make it small for local search),
    and tr1 and tr2 are randomly-generated trees."""

    # choosing maxdepth 2 for these trees
    tr1 = grow(2, random)
    tr2 = grow(2, random)
    return ['+', t, ['*', ms, ['-', tr1, tr2]]]

def hillclimb(fitness_fn, n_evals=2000, popsize=100):
    """Hill-climbing optimisation. """

    # Generate an initial solution
    t = grow(2, random)
    # Get its fitness value (ignore its semantics, even though they
    # are returned also)
    fnt = make_fn(t)
    ft, _ = fitness_fn.get_semantics(fnt)
    
    for i in xrange(n_evals):
        # mutation step size
        ms = np.random.normal()

        # Mutate and get fitness of child
        s = semantic_geometric_mutate(t, ms)
        fns = make_fn(s)
        fs, _ = fitness_fn.get_semantics(fns)

        # Keep the child only if better
        if fs < ft:
            t, ft, fnt = s, fs, fns
            
        # Simulate generations by printing after popsize have been
        # evaluated, even though there is no population in this
        # hillclimbing algorithm.
        if i % popsize == popsize - 1: 
            print("%d %f %f %s" % (i, ft, fitness_fn.test(fnt), str(t)))
        
if __name__ == "__main__":
    # srff is a symbolic regression problem imported from fitness.py
    # -- it's the Vanneschi et al bioavailability one (unless
    # fitness.py has been edited recently). you have to run the
    # get_data.py script first to download the data.
    hillclimb(srff, 200, 20)
