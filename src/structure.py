#!/usr/bin/env python

import sys, time, random
import numpy as np
import nonrandom
from copy import deepcopy
from variga import xover, Individual
from variga import fitness_idx, used_idx, genome_idx, phenotype_idx, readable_phenotype_idx, semantics_idx


# note xover takes parent inds and returns genomes
# onepoint_mutate takes a genome and modifies it (and returns it)
def onepoint_mutate(g, used):
    mut_pt = random.randint(0, used-1)
    g[mut_pt] = random.randint(0, MAXV-1)
    return g

def generate_random_pairs(n):
    for i in range(n):
        LEN = random.randint(MINLEN, MAXLEN)
        g = [random.randint(0, MAXV-1) for i in range(LEN)]
        gind = Individual(g)
        if gind[phenotype_idx] is None:
            continue
        h = [random.randint(0, MAXV-1) for i in range(LEN)]
        hind = Individual(h)
        if hind[phenotype_idx] is None:
            continue
        yield (gind, hind)
    
def generate_mutation_pairs(n):
    for i in range(n):
        LEN = random.randint(MINLEN, MAXLEN)
        g = [random.randint(0, MAXV-1) for i in range(LEN)]
        gind = Individual(g)
        if gind[phenotype_idx] is None:
            continue
        h = onepoint_mutate(g[:], min(gind[used_idx], len(g)))
        hind = Individual(h)
        if hind[phenotype_idx] is None:
            continue
        yield (gind, hind)

def generate_crossover_pairs(n):
    for i in range(n):
        LEN = random.randint(MINLEN, MAXLEN)
        g = [random.randint(0, MAXV-1) for i in range(LEN)]
        gind = Individual(g)
        if gind[phenotype_idx] is None:
            continue
        LEN = random.randint(MINLEN, MAXLEN)
        h = [random.randint(0, MAXV-1) for i in range(LEN)]
        hind = Individual(h)
        if hind[phenotype_idx] is None:
            continue
        children = xover(gind, hind)
        cind0 = Individual(children[0])
        cind1 = Individual(children[1])
        if cind0[phenotype_idx] is not None:
            yield (gind, cind0)
            yield (hind, cind0)
        if cind1[phenotype_idx] is not None:
            yield (gind, cind1)
            yield (hind, cind1)
        

def distances(g, h):
    # g and h are individuals
    return (levenshtein(g[genome_idx], h[genome_idx]),
            PHENOTYPE_DISTANCE(g[readable_phenotype_idx], h[readable_phenotype_idx]),
            SEMANTIC_DISTANCE(g[semantics_idx], h[semantics_idx]),
            abs(g[fitness_idx] - h[fitness_idx]))

def levenshtein(a,b):
    """Calculates the Levenshtein distance between a and b. Copied
    from [http://hetland.org/coding/python/levenshtein.py]"""
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n
        
    current = range(n+1)
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)
            
    return current[n]

def hamming_distance(g, h):
    # g and h are genomes
    d = 0
    for gi, hi in zip(g, h):
        if gi != hi:
            d += 1
    return d
    
def main():
    from sys import argv
    print levenshtein(argv[1],argv[2])
    print levenshtein([4, 5, 6, 7], [5, 6, 7])

if __name__ == "__main__":
    main()
