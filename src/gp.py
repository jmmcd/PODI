#!/usr/bin/env python

# This module is for doing standard GP stuff. There are standard
# operators and the semantic geometric mutation. They make random
# choices, just like standard GP, and they don't use an NRNG to do so.
# However, PODI also uses some functions from here.

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

mutation_prob = 0.01

# srff = fitness.benchmarks("pagie_2d")
# srff = fitness.benchmarks("vanneschi_bioavailability")
# FIXME this stuff shouldn't be done at module level
srff = fitness.benchmarks("evocompetitions_2010")

# pdff_n_samples = 100
# pdff = fitness.ProbabilityDistributionFitnessFunction(
#     np.linspace(0.0, 1.0, pdff_n_samples), pdff_n_samples)

# variables = ["x", "y"]
variables = ["x" + str(i) for i in range(srff.arity)]

# consider allowing all the distributions here
# [http://docs.scipy.org/doc/numpy/reference/routines.random.html] as
# primitives in the pdff. Consider at least normal, Poisson, beta,
# uniform, exponential, lognormal, weibull.

# For now, RAND just gives a uniform.
# variables = ["RAND"] # see evaluate() above

constants = [0.1, 0.2, 0.3, 0.4, 0.5]
variables = variables + constants
fns = {"+": 2, "-": 2, "*": 2, "/": 2, "sin": 1, "cos": 1, "square": 1}

# SIF is the soft-if function from: Will Smart and Mengjie Zhang,
# Using Genetic Programming for Multiclass Classification by
# Simultaneously Solving Component Binary Classification Problems
# http://www.mcs.vuw.ac.nz/comp/Publications/archive/CS-TR-05/CS-TR-05-1.pdf
def SIF(x, y, z):
    return (y/(1.0+e**(2*x)) + z/(1.0+e**(-2*x)))

# AQ is the analytic quotient from: Ji Ni and Russ H. Drieberg and
# Peter I. Rockett, "The Use of an Analytic Quotient Operator in
# Genetic Programming", IEEE Transactions on Evolutionary Computation
def AQ(x, y):
    return x/sqrt(1.0+y*y)


def _evaluate(t, x):
    if type(t) == str:
        # it's a single string
        if t[0] == "x":
            idx = int(t[1:])
            return x[idx] 
        elif t == "x": return x[0]
        elif t == "y": return x[1]
        # special var generates a random var uniformly in [0, 1].
        elif t == "RAND": return np.random.random(len(x[0]))
        else:
            try:
                return np.ones(len(x[0])) * float(t)
            except ValueError:
                raise ValueError("Can't interpret " + t)
    elif type(t) in (int, float, np.float64):
        return np.ones(len(x[0])) * t
    else:
        assert(type(t) == list)
        # it's a list: take t[0] and decide what to do
        if   t[0] == "+": return add(evaluate(t[1], x), evaluate(t[2], x))
        elif t[0] == "-": return subtract(evaluate(t[1], x), evaluate(t[2], x))
        elif t[0] == "*": return multiply(evaluate(t[1], x), evaluate(t[2], x))
        elif t[0] == "/":
            try:
                return divide(evaluate(t[1], x), evaluate(t[2], x))
            except FloatingPointError:
                return evaluate(t[1], x)
        elif t[0] == "sin": return sin(evaluate(t[1], x))
        elif t[0] == "cos": return cos(evaluate(t[1], x))
        elif t[0] == "square": return square(evaluate(t[1], x))
        elif t[0] == "AQ": return AQ(evaluate(t[1], x), evaluate(t[2], x))
        elif t[0] == "SIF": return SIF(evaluate(t[1], x), evaluate(t[2], x), evaluate(t[3], x))
        else:
            raise ValueError("Can't interpret " + t[0])

# evaluate = MemoizeMutable(_evaluate)
evaluate = _evaluate

def make_fn(t):
    return lambda x: evaluate(t, x)

def isatom(t):
    return (isinstance(t, str) or isinstance(t, float)
            or isinstance(t, int))

def traverse(t, path=None):
    """Depth-first traversal of the tree t, yielding at each step the
    node, the subtree rooted at that node, and the path. The path
    passed-in is the "path so far"."""
    if path is None: path = tuple()
    yield t[0], t, path + (0,)
    for i, item in enumerate(t[1:], start=1):
        if isatom(item):
            yield item, item, path + (i,)
        else:
            for s in traverse(item, path + (i,)):
                yield s

def place_subtree_at_path(t, path, st):
    """Place subtree st into tree t at the given path. Cannot
    correctly place a single node at the root of t."""
    if path == (0,):
        if isatom(st):
            raise ValueError("Cannot place a single node at the root")
        else:
            t[:] = st
            return
    if path[-1] == 0:
        # Trying to overwrite a subtree rooted at the node given by
        # path: We have to go back up one
        path = path[:-1]
    ptr = get_subtree(t, path[:-1])
    ptr[path[-1]] = st # ...because it's the final index

def get_node(t, path):
    """Given a tree and a path, return the node at that path."""
    s = get_subtree(t, path)
    if isatom(s):
        return s
    else:
        return s[0]

def get_subtree(t, path):
    """Given a tree and a path, return the subtree at that path."""
    for item in path:
        t = t[item]
    return t

def tree_depth(t):
    """The depth of a tree is the maximum depth of any of its nodes.
    FIXME if a bare node is passed-in, this will crash."""
    d = 0
    for nd, st, path in traverse(t):
        dn = depth(path)
        if dn > d: d = dn
    return d

def depth(path):
    """The depth of any node is the number of nonzero elements in its
    path."""
    return len([el for el in path if el != 0])

def grow(st_maxdepth, rng):
    pTerminal = 0.2 # FIXME not sure where/how to parameterise this
    if st_maxdepth == 0 or rng.random() < pTerminal:
        return rng.choice(variables)
    else:
        nd = rng.choice(fns.keys())
        return [nd] + [grow(st_maxdepth-1, rng) for i in range(fns[nd])]

def make_zssNode_from_tuple(t):
    """Convert my tuple representation into the Node object
    representation used in zss."""
    n = Node(t[0])
    for s in t[1:]:
        n.addkid(make_zssNode_from_tuple(s))
    return n

def tree_distance(t, s):
    # print(t)
    # print(s)
    assert(t is not None)
    assert(s is not None)
    tN = make_zssNode_from_tuple(t)
    sN = make_zssNode_from_tuple(s)
    return zss.compare.distance(tN, sN)

def semantic_distance(v, u):
    """Semantic distance between two semantics objects. Assume that
    inputs are vectors of numbers, ie values at SR-style fitness
    cases. Distance is just Euclidean distance."""
    try:
        return np.linalg.norm(v - u)
    except FloatingPointError as e:
        print("FPE in distances", e)
        return 0.0
    except TypeError as e:
        print("TypeError in distances", e)
        return 0.0

def iter_len(iter, filter=lambda x: True):
    return sum(1 for x in iter if filter(x))

def crossover(t1, t2):
    """Subtree crossover. Unused for now."""
    t1 = copy.deepcopy(t1)
    t2 = copy.deepcopy(t2)

    # Choose crossover points s1, s2 -- don't crossover at root
    n1 = iter_len(traverse(t1))
    s1 = random.randint(1, n1-1) 
    i1 = 0
    n2 = iter_len(traverse(t2))
    s2 = random.randint(1, n2-1)
    i2 = 0
    # Find crossover points
    for nd1, st1, path1 in traverse(t1):
        if i1 == s1:
            for nd2, st2, path2 in traverse(t2):
                if i2 == s2:
                    place_subtree_at_path(t1, path1, st2)
                    place_subtree_at_path(t2, path2, st1)
                    break
                i2 += 1
            break
        i1 += 1
    return t1, t2

def subtree_mutate(t, st_maxdepth=3):
    """Mutate a tree by growing a new subtree at a random location."""
    
    # this deepcopy is necessary, but will be slow
    t = copy.deepcopy(t)
     
    n = iter_len(traverse(t))
    s = random.randint(0, n-1)

    if s == 0:
        # mutate at root: return an entirely new tree. can't be done
        # using place_subtree_at_path as below
        return grow(st_maxdepth, random)
    else:
        # don't mutate at root
        i = 0
        for nd, st, path in traverse(t):
            if i == s:
                place_subtree_at_path(t, path, grow(st_maxdepth, random))
                break
            i += 1
        return t
    
def point_mutate(t, p=mutation_prob):
    """Point mutation of a tree. Traverse the tree. For each node,
    with low probability, replace it with another node of same
    arity."""
    
    # this deepcopy is necessary, but will be slow
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
                options = [term for term in variables if term != nd]
                if options:
                    place_subtree_at_path(t, path, random.choice(options))
    return t
            
            
def semantic_geometric_mutate(t, ms=0.01, st_maxdepth=3):
    """Semantic geometric mutation as defined by Moraglio et al:

    tm = t + ms * (tr1 - tr2)
    
    where ms is the mutation step (make it small for local search),
    and tr1 and tr2 are randomly-generated trees."""

    tr1 = grow(st_maxdepth, random)
    tr2 = grow(st_maxdepth, random)
    return ['+', t, ['*', ms, ['-', tr1, tr2]]]

def semantic_geometric_mutate_differentiate(t, fitness_fn, st_maxdepth=3):
    """Semantic geometric mutation with differentiation:

    tm = t + ms * tr
    
    where tr is a randomly-generated tree and ms is the mutation step,
    which can be negative, found by diffentiating the new error
    RMSE(y, t + ms * tr) with respect to ms. To make this work the
    mutation operator needs to be able to evaluate, so we have to pass
    in the fitness function.

    The optimum mutation step ms is such that RMSE is minimised. But
    minimising RMSE is equivalent to minimising mean square error
    (MSE):

    MSE = mean((y - (t + ms*tr))**2)
        = mean(((y-t) - ms*tr)**2)
        = mean((y-t)**2 - 2*(y-t)*ms*tr + ms**2*tr**2)

    Differentiate wrt ms:
    
    d(MSE)/d(ms) = mean(-2*(y-t)*tr + 2*ms*tr**2)
                 = -2*mean((y-t)*tr) + 2*ms*mean(tr**2)
                 
    This is zero when:
    
    2*mean((y-t)*tr) = 2*ms*mean(tr**2)
	
    Therefore the optimum ms is:
    
    ms = mean((y-t)*tr) / mean(tr**2)"""


    # Generate a tree tr and make sure it won't return all zeros,
    # which would trigger a divide-by-zero. Start with all zeros to
    # get into the while loop.
    tr_out = np.zeros(2)
    while np.mean(tr_out**2) < 0.000001:
        tr = grow(st_maxdepth, random)
        tr_out = fitness_fn.get_semantics(make_fn(tr))[1]

    t_out = fitness_fn.get_semantics(make_fn(t))[1] # should be cached already
    y = fitness_fn.train_y
    # formula from above comment
    ms = np.mean((y-t_out)*tr_out) / np.mean(tr_out**2)

    # TODO if ms is close to zero, we could reject the step and try
    # again, for a kind of ad-hoc regularisation. The threshold could
    # be annealed during the run, perhaps. For now, just accept the
    # step regardless.
    
    return ['+', t, ['*', ms, tr]]


def hillclimb(fitness_fn, mutation_type="optimal_ms",
              ngens=2000, popsize=1, print_every=200, st_maxdepth=3):
    """Hill-climbing optimisation. """
    assert(print_every % popsize == 0)
    evals = 0
    
    print("# generation evaluations best_fitness best_test_fitness best_phenotype")
    # Generate an initial solution
    t = grow(st_maxdepth, random)
    # Get its fitness value (ignore its semantics, even though they
    # are returned also)
    fnt = make_fn(t)
    ft, _ = fitness_fn.get_semantics(fnt)
    
    for gen in xrange(ngens):

        # make a lot of new individuals by mutation
        if mutation_type == "GSGP-optimal-ms":
            # Mutate and differentiate to get the best possibility
            s = [semantic_geometric_mutate_differentiate(t, fitness_fn, st_maxdepth)
                 for i in range(popsize)]
                
        elif mutation_type == "GSGP":
            # mutation step size randomly chosen
            s = [semantic_geometric_mutate(t, np.random.normal(), st_maxdepth)
                 for i in range(popsize)]

        elif mutation_type == "GP":
            s = [subtree_mutate(t, st_maxdepth)
                 for i in range(popsize)]
        else:
            raise ValueError("Unknown mutation type " + mutation_type)

        # test the new individuals and keep only the single best
        # 
        for si in s:
            # Evaluate child
            fnsi = make_fn(si)
            fsi, _ = fitness_fn.get_semantics(fnsi)

            # Keep the child only if better
            if fsi < ft:
                t, ft, fnt = si, fsi, fnsi

        evals += popsize
        if evals % print_every == 0:
            print("%d %d %f %f : %s" % (gen, evals, 
                                        ft, fitness_fn.test(fnt), str(t)))
        
if __name__ == "__main__":
    # srff is a symbolic regression problem -- defined at top of file.
    # you might have to run the get_data.py script first to download
    # data.
    hillclimb(srff, "GSGP-optimal-ms", 5, 1, 1, 3)
