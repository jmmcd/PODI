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

# TODO Move all the GP code from bubble_down into this file
from bubble_down import mknd, mkst, depth, get_node, get_subtree, traverse, make_fn, evaluate, grow, srff, place_subtree_at_path, tree_depth
from bubble_down import vars, fns

mutation_prob = 0.01

def iter_len(iter):
    return sum(1 for _ in iter)

def subtree_mutate(t):
    # FIXME this deepcopy is probably necessary, but will probably be
    # slow
    t = copy.deepcopy(t)
    
    n = iter_len(traverse(t))
    s = random.randint(1, n-1) # don't mutate at root
    i = 0
    for nd, st, path in traverse(t):
        if i == s:
            # stn = iter_len(st) # nnodes in subtree
            # FIXME how should we set limit on size of tree given by
            # grow? For now, setting depth limit to 3
            place_subtree_at_path(t, path, grow(3, random))
            break
        i += 1
    return t
    
def point_mutate(t, p=mutation_prob):
    """Point mutation of a tree. Traverse the tree. For each node,
    with low probability, replace it with another node of same arity.
    FIXME would be easy to use grow and place_subtree_at_path to get
    subtree mutation."""
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
                    place_subtree_at_path(t, path, random.choice(options))
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

    print("# generation evaluations best_fitness best_test_fitness best_phenotype")
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
            print("%d %d %f %f : %s" % (i % popsize, i,
                                        ft, fitness_fn.test(fnt), str(t)))
        
if __name__ == "__main__":
    # srff is a symbolic regression problem imported from fitness.py
    # -- it's the Vanneschi et al bioavailability one (unless
    # fitness.py has been edited recently). you have to run the
    # get_data.py script first to download the data.
    hillclimb(srff, 200, 20)
