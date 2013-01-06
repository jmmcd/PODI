#!/usr/bin/env python

from grammar import Grammar
import variga
import fitness

def random_dt(random, grammar, s=None):
    """Recursively create a random derivation tree given a start
    symbol and a grammar. By plugging in variga, we mimic GE with the
    bucket rule."""
    if s is None:
        s = grammar.start_rule[0]
    elif s in grammar.terminals:
        return s
    rule = grammar.rules[s]
    if len(rule) > 1:
        prod = random.choice(rule)
    else:
        prod = rule[0]
    return [s] + [random_dt(random, grammar, s[0]) for s in prod]

def random_dt_mod(random, grammar, s=None):
    """Recursively create a random derivation tree given a start
    symbol and a grammar. This version aims to mimic the GE mod rule."""
    if s is None:
        s = grammar.start_rule[0]
    elif s in grammar.terminals:
        return s
    rule = grammar.rules[s]
    if len(rule) > 1:
        codon = random.randint(0, MAX_CODON)
        idx = codon % len(rule)
        prod = rule[idx]
    else:
        prod = rule[0]
    return [s] + [random_dt_mod(random, grammar, s[0]) for s in prod]

def random_str_mod(random, grammar, s=None):
    """Recursively derive a random string tree given a start symbol
    and a grammar. Don't create a dt. This version aims to mimic the
    GE mod rule."""
    if s is None:
        s = grammar.start_rule[0]
    elif s in grammar.terminals:
        return s
    rule = grammar.rules[s]
    if len(rule) > 1:
        codon = random.randint(0, MAX_CODON)
        idx = codon % len(rule)
        prod = rule[idx]
    else:
        prod = rule[0]
    return "".join([random_str_mod(random, grammar, s[0]) for s in prod])

def derived_str(dt, grammar):
    """Get the derived string."""
    return "".join([s[0] for s in traverse(dt) if s[0] in grammar.terminals])

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

# grammar = Grammar("grammars/symbolic_regression_2d.bnf")
grammar = Grammar("grammars/sr_2d_ne_test.bnf")
srff = fitness.benchmarks()["pagie_2d"]
MAX_CODON = 127
def generate(random):
    return random_str_mod(random, grammar)
def success(err):
    return False # let's just keep running so all runs are same length
variga.GENERATE = generate
variga.FITNESS = srff
variga.SUCCESS = success
variga.POPSIZE = 1000
variga.GENERATIONS = 40
variga.PMUT = 0.01
variga.CROSSOVER_PROB = 0.7
variga.MINLEN = 100 # ponyge uses 100 for all initialisation, no min/maxlen
variga.MAXLEN = 100
variga.MAXIMISE = False
variga.ELITE = 1
variga.TOURNAMENT_SIZE = 3
variga.WRAPS = 1
variga.main()
