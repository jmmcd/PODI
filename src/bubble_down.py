#!/usr/bin/env python

import os
import sys
import random
import collections
import zss # git clone https://github.com/timtadh/zhang-shasha.git
from zss.test_tree import Node
import numpy as np
from numpy import add, subtract, multiply, divide, sin, cos, exp, log, power, square
np.seterr(all='raise')

import structure
import fitness
import variga

# uncomment these lines in order to get deterministic tree generation
# import nonrandom
# random = nonrandom.NonRandom([999999999, 0, 17] * 100)

def evaluate(t, x):
    if type(t) == type(""):
        # it's a single string
        if t[0] == "x":
            idx = int(t[1:])
            return x[idx] 
        elif t == "x": return x[0]
        elif t == "y": return x[1]
        # special var generates a random var uniformly in [0, 1].
        # FIXME this will be slow -- would rather generate a column of
        # random numbers, allowing np broadcasting throughout
        # evaluate(), but I don't know how to get the right length
        # from within this function.
        elif t == "RAND": return np.random.random()
        else:
            try:
                return float(t)
            except ValueError:
                raise ValueError("Can't interpret " + t)
    else:
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
        else:
            raise ValueError("Can't interpret " + t[0])

def make_fn(t):
    return lambda x: evaluate(t, x)

def traverse(t, path=None):
    """Depth-first traversal of the tree t, yielding at each step the
    node, the subtree rooted at that node, and the path. The path
    passed-in is the "path so far"."""
    if path is None: path = tuple()
    yield t[0], t, path + (0,)
    for i, item in enumerate(t[1:], start=1):
        if isinstance(item, str):
            yield item, item, path + (i,)
        else:
            for s in traverse(item, path + (i,)):
                yield s

def get_node(t, path):
    """Given a tree and a path, return the node at that path."""
    s = get_subtree(t, path)
    if isinstance(s, str):
        return s
    else:
        return s[0]

def get_subtree(t, path):
    """Given a tree and a path, return the subtree at that path."""
    for item in path:
        t = t[item]
    return t

def depth(path):
    """The depth of any node is the number on nonzero elements in its
    path."""
    return len([el for el in path if el != 0])

def mknd(rng):
    """Make a node, consisting of a label, weights, and a bias
    term."""
    lbl = rng.choice(fns.keys())
    wts = sorted([rng.random() for i in range(fns[lbl] - 1)])
    bias = rng.random()
    return (lbl, wts, bias)

def mkst(rng):
    """Make a subtree, consisting of a node and the appropriate number
    of null children."""
    nd = mknd(rng)
    return [nd] + [None] * fns[nd[0]]

def choose_child(weights, val):
    """Calculate an index, ie the slot where val fits, given val (in
    [0, 1]) and weights (sorted numbers in [0, 1])"""
    for i, wt in enumerate(weights):
        if val < wt:
            return i
        return len(weights)
    return 0 # because weights is of len 0

def bubble_down_minn_maxn(minn, maxn, rng):
    """Generate a tree whose node-count is between the bounds minn and
    maxn (or exceeds maxn by up to (maximum arity - 1). We can fix
    bounds in advance, and let evolution control tree size."""
    n = rng.randint(minn, maxn)
    return bubble_down(n, rng)    

def bubble_down(n, rng):
    """Generate a tree of the desired node-count (can exceed it by up
    to (maximum arity - 1). See
    http://en.wikipedia.org/wiki/Random_binary_tree.

    The binary-only version of this algorithm works by analogy with a
    binary search tree. Each new node gets a random number and is then
    filtered down the tree by comparing that number with those of the
    existing nodes. Our version gives (arity - 1) random weights to
    each node, as well as a separate bias value. Each new nonterminal
    node is filtered down the tree by comparing its bias value with
    the weights of the existing nodes. Once a sufficient number of
    nonterminals have been added, we stop that process and fill in
    terminals along the frontier, stripping out the now unneeded
    weights and bias values.

    The algorithm is probably O(n logn) or so, since each node must be
    filtered down to at most the maximum depth of the tree, which is
    roughly log n."""

    if n <= 1:
        return [rng.choice(vars)]
    t = mkst(rng)
    node_cnt = len(t)
    max_depth = 0

    # Keep adding nodes until the number of internal nodes + null leaf
    # nodes is sufficient
    while node_cnt < n:
        st = mkst(rng)
        node_cnt += len(st) - 1
        # print("at the top: t = " + str(t) + " st = " + str(st) + " node_cnt = " + str(node_cnt))

        current = t
        path = tuple()
        # take this subtree st and bubble it down.
        while current:
            # print("bubbling: current = " + str(current))
            current_wts = current[0][1] # [0] gets node, [1] gets wts
            st_bias = st[0][2] # [0] gets node, [2] gets bias
            child_idx = choose_child(current_wts, st_bias)
            assert(child_idx is not None)
            current = current[1+child_idx] # +1 because subtree root is at 0
            path = path + (1+child_idx,) # +1 again
        accessor = get_subtree(t, path[:-1]) # don't use element -1...
        accessor[path[-1]] = st # ...because it's the final index
        d = depth(path)
        if d > max_depth:
            max_depth = d

    # since we already know node_cnt and depth, useful to return them
    # note max depth gets +1 because we have yet to add leaves.
    return add_leaves_remove_annotations(t, rng), node_cnt, max_depth+1

def add_leaves_remove_annotations(t, rng):
    result = [t[0][0]] # first [0] gets node, second [0] gets label
    for item in t[1:]:
        if item is None:
            result.append(rng.choice(vars)) # fill in a leaf
        else:
            result.append(add_leaves_remove_annotations(item, rng))
    return result


def grow(maxdepth, rng):
    if maxdepth == 0 or rng.random() < pTerminal:
        return rng.choice(vars)
    else:
        nd = rng.choice(fns.keys())
        return [nd] + [grow(maxdepth-1, rng) for i in range(fns[nd])]

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
    """FIXME Inputs are vectors of numbers, ie values at fitness
    cases. Distance is just Euclidean distance."""
    try:
        return np.linalg.norm(v - u)
    except FloatingPointError as e:
        print("FPE in distances", e)
        return 0.0
    except TypeError as e:
        print("TypeError in distances", e)
        return 0.0

def generate_bubble_down_tree_and_fn_minn_maxn(minn, maxn, rng):
    t, d, n = bubble_down_minn_maxn(minn, maxn, rng)
    return t, make_fn(t)

def generate_bubble_down_tree_and_fn(rng):
    t, d, n = bubble_down(30, rng)
    return t, make_fn(t)

def generate_grow_tree_and_fn(rng):
    t = grow(6, rng)
    return t, make_fn(t)

def success(err):
    return False # let's just keep running so all runs are same length

def study_structure(basename, rep="bubble_down"):
    """Use structure.py to investigate the structure of the tree-based
    GP space: can do bubble-down or grow algorithm."""
    structure.MINLEN = 100
    structure.MAXLEN = 100
    structure.SEMANTIC_DISTANCE = semantic_distance
    structure.PHENOTYPE_DISTANCE = tree_distance
    structure.FITNESS = srff
    structure.CROSSOVER_PROB = 1.0
    structure.MAXV = sys.maxint
    structure.WRAPS = 0

    if rep == "bubble_down":
        structure.GENERATE = generate_bubble_down_tree_and_fn
    elif rep == "grow":
        structure.GENERATE = generate_grow_tree_and_fn
    else:
        raise ValueError
    structure.MAXIMISE = False
    n = 10000
    print(rep)
    print("random")
    with open(os.path.join(basename, rep, "random_distances.dat"), "w") as outfile:
        total_count = 0
        neutral_count = 0
        for g, h in structure.generate_random_pairs(n):
            total_count += 1
            ds = structure.distances(g, h)
            if ds[1] > 0:
                outfile.write("%d %d %f %f\n" % (ds))
            else:
                neutral_count += 1
        print("random attempted trials %d, valid %d, neutral %d" % (
                n, total_count, neutral_count))
    print("mutation")
    with open(os.path.join(basename, rep, "mutation_distances.dat"), "w") as outfile:
        total_count = 0
        neutral_count = 0
        for g, h in structure.generate_mutation_pairs(n):
            total_count += 1
            ds = structure.distances(g, h)
            if ds[1] > 0:
                outfile.write("%d %d %f %f\n" % (ds))
            else:
                neutral_count += 1
        print("mutation attempted trials %d, valid %d, neutral %d" % (
                n, total_count, neutral_count))
    print("crossover")
    with open(os.path.join(basename, rep, "crossover_distances.dat"), "w") as outfile:
        total_count = 0
        neutral_count = 0
        for g, c in structure.generate_crossover_pairs(n):
            total_count += 1
            ds = structure.distances(g, c)
            if ds[1] > 0:
                outfile.write("%d %d %f %f\n" % (ds))
            else:
                neutral_count += 1
        print("crossover attempted trials %d, valid %d, neutral %d" % (
                n, total_count, neutral_count))

def test():
    t, dep, nnodes = bubble_down(random.randint(5, 19), random)
    s, dep, nnodes = bubble_down(random.randint(10, 15), random)
    print(t)
    print(s)
    tN = make_zssNode_from_tuple(t)
    sN = make_zssNode_from_tuple(s)
    print(tN)
    print(sN)
    print(zss.compare.distance(tN, sN))
    print(zss.compare.distance(tN, tN))
    print(zss.compare.distance(sN, sN))

    print("---")
    cases = [(x, y) for x in range(4) for y in range(4)]
    print(cases)
    cases = zip(*cases)
    print(cases)
    tf = make_fn(t)
    print(tf(cases))

def test_grow():
    for i in range(7):
        print(grow(i, random))

def run(fitness_fn, rep="bubble_down"):
    variga.MINLEN = 100
    variga.MAXLEN = 100
    variga.PHENOTYPE_DISTANCE = tree_distance
    variga.FITNESS = fitness_fn
    if rep == "bubble_down":
        variga.GENERATE = generate_bubble_down_tree_and_fn
    elif rep == "grow":
        variga.GENERATE = generate_grow_tree_and_fn
    else:
        raise ValueError
    variga.MAXIMISE = False
    variga.SUCCESS = success
    variga.POPSIZE = 40
    variga.GENERATIONS = 10
    variga.PMUT = 0.01
    variga.CROSSOVER_PROB = 0.7
    variga.ELITE = 1
    variga.TOURNAMENT_SIZE = 3
    variga.WRAPS = 1
    variga.main()

# srff = fitness.benchmarks()["pagie_2d"]
srff = fitness.benchmarks()["vanneschi_bioavailability"]

pdff_n_samples = 10
pdff = fitness.ProbabilityDistributionFitnessFunction(
    np.linspace(0.0, 1.0, pdff_n_samples), pdff_n_samples)

# vars = ["x", "y"]
vars = ["x" + str(i) for i in range(srff.arity)]

# consider allowing all the distributions here
# [http://docs.scipy.org/doc/numpy/reference/routines.random.html] as
# primitives in the pdff. For now, RAND just gives a uniform.
vars = ["RAND"] # see evaluate() above
consts = ["0.1", "0.2", "0.3", "0.4", "0.5"]
vars = vars + consts
fns = {"+": 2, "-": 2, "*": 2, "/": 2, "sin": 1, "cos": 1, "square": 1}
# fns = {"+": 2, "-": 2, "*": 2, "/": 2}

# FIXME try adding the soft-if function from: Using Genetic
# Programming for Multiclass Classification by Simultaneously Solving
# Component Binary Classification Problems, Will Smart, Mengjie Zhang

# FIXME try adding the no-exceptions division which featured in a
# TransEC article recently.


pTerminal = 0.2 # used in grow algorithm


if __name__ == "__main__":
    if sys.argv[1] == "test":
        test()
    elif sys.argv[1] == "test_grow":
        test_grow()
    elif sys.argv[1] == "grow_structure":
        study_structure(sys.argv[2], "grow")
    elif sys.argv[1] == "bubble_down_structure":
        study_structure(sys.argv[2], "bubble_down")
    elif sys.argv[1] == "run_bubble_down":
        run(srff, "bubble_down")
    elif sys.argv[1] == "run_grow":
        run(srff, "grow")
    elif sys.argv[1] == "run_bubble_down_prob":
        run(pdff, "bubble_down")
    elif sys.argv[1] == "run_grow_prob":
        run(pdff, "grow")
    else:
        print("Usage: <test|structure>")
        print(sys.argv)
