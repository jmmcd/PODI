#!/usr/bin/env

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

from bubble_down import mknd, mkst, depth, get_node, get_subtree, traverse, make_fn, evaluate, grow

from bubble_down import vars, fns

pMut = 0.01

def mutate(t):
    """Point mutation of a tree. Traverse the tree. For each node,
    with low probability, replace it with another node of same
    arity."""
    t = copy.deepcopy(t) # FIXME this is probably necessary, will probably be slow
    for nd, st, path in traverse(t):
        print(nd, st, path)
        if random.random() < pMut:
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
                    
            
            
def semantic_mutate(t):
    # tm = t + ms * (tr1 - tr2)
    # where ms is the mutation step (make it small for local search)
    # tr1 and tr2 are randomly-generated trees (I will make them small)

    ms = 0.1
    tr1 = grow(2, random)
    tr2 = grow(2, random)
    return ['+', t, ['*', ms, ['-', tr1, tr2]]]
