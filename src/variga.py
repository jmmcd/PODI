#!/usr/bin/env python

import sys, time, random
import numpy as np
import nonrandom
from copy import deepcopy

fitness_idx, used_idx, genome_idx, phenotype_idx = range(4)
    
# If you try to compare individuals directly, using tuple comparison,
# you can get in trouble when the genomes have different lengths and
# the fitness and used values are the same, because max() will compare
# the next element of the tuples, the genomes, and max() on numpy
# arrays will try to produce a new array of the pointwise max
# values. So we only use the fitness. High fitness is good for
# MAXIMISE, else low fitness.
def ind_compare(x):
    if MAXIMISE:
        return x[fitness_idx]
    else:
        return -x[fitness_idx]

def Individual(genome):
    if genome is None:
        LEN = random.randint(MINLEN, MAXLEN)
        genome = [random.randint(0, MAXV-1) for i in range(LEN)]
    nr = nonrandom.NonRandom(genome, maxval=MAXV, wraps=WRAPS)
    try:
        phenotype = GENERATE(nr)
        fitness = FITNESS(phenotype)
    except StopIteration:
        phenotype = None
        if MAXIMISE:
            fitness = -float("inf")
        else:
            fitness = float("inf")
    used = nr.used
    return (fitness, used, genome, phenotype)

# Onepoint crossover FIXME change to twopoint?
def xover(a, b):
    g, h = a[genome_idx], b[genome_idx]
    if random.random() < CROSSOVER_PROB:
        # -1 to get last index in array; min() in case of wraps: used > len
        max_g, max_h = min(len(g)-1, a[used_idx]), min(len(h)-1, b[used_idx])
        pt_g, pt_h = random.randint(1, max_g), random.randint(1, max_h)
        c = g[:pt_g] + h[pt_h:]
        d = h[:pt_h] + g[pt_g:]
        return c, d
    else:
        return g[:], h[:]

# Per-gene bit-flip mutation FIXME allow insert/delete? -- would be
# more suited to the ripple effect, in a way.  FIXME could switch to
# float rep and use gaussian mutation could even use gaussian-style
# mutation on integers...
def mutate(g):
    for pt in range(len(g)):
        if random.random() < PMUT:
            g[pt] = random.randint(0, MAXV-1)
    return g

# Print statistics, and return True if we have succeeded already.
def stats(pop, gen):
    best = max(pop, key=ind_compare)
    valids = [i for i in pop if i[phenotype_idx] is not None]
    ninvalids = len(pop) - len(valids)
    if len(valids) == 0:
        fitness_vals = np.array([0])
        used_vals = np.array([0])
    else:
        fitness_vals = np.array([i[fitness_idx] for i in valids])
        used_vals = np.array([i[used_idx] for i in valids])
    len_vals = np.array([len(i[genome_idx]) for i in pop])
    meanfit = np.mean(fitness_vals)
    sdfit = np.std(fitness_vals)
    meanused = np.mean(used_vals)
    sdused = np.std(used_vals)
    meanlen = np.mean(len_vals)
    sdlen = np.std(len_vals)
    if gen == 0:
        print("# generation evaluations best_fitness best_used_codons " +
              "best_test_fitness " +
              "mean_fitness stddev_fitness " + 
              "mean_used_codons stddev_used_codons " +
              "mean_genome_length stddev_genome_length " +
              "number_invalids best_phenotype")
    print("{0} {1} {2} {3} {4} {5:.2f} {6:.2f} {7:.2f} {8:.2f} {9:.2f} {10:.2f} {11} : {12}"
          .format(gen, POPSIZE * gen,
                  best[fitness_idx], best[used_idx],
                  FITNESS.test(best[phenotype_idx]),
                  meanfit, sdfit, meanused, sdused,
                  meanlen, sdlen, ninvalids, 
                  best[phenotype_idx]))
    return SUCCESS(best[fitness_idx])

# Use many tournaments to get parents
def tournament(items):
    while True:
        candidates = random.sample(items, TOURNAMENT_SIZE)
        yield max(candidates, key=ind_compare)

# Run one generation
def step(pop):
    pop.sort(key=ind_compare)
    assert ELITE < POPSIZE
    elite = pop[-ELITE:] # best inds: how many? ELITE
    
    # crossover: pass inds, get new genomes
    newpop = []
    parents = SELECTION(pop)
    while len(newpop) < POPSIZE:
        for child_genome in xover(parents.next(), parents.next()):
            if len(newpop) < POPSIZE:
                newpop.append(child_genome)
        
    # mutation: pass genomes, they'll be changed
    for genome in newpop:
        mutate(genome)

    # grow up: turn genomes into individuals
    newpop = [Individual(g) for g in newpop]
    
    # elite: replace worst
    newpop.sort(key=ind_compare)
    newpop[:ELITE] = elite
    return newpop
    
def main(seed=None):
    if seed is not None:
        random.seed(seed)
    pop = [Individual(None) for i in range(POPSIZE)]
    for gen in range(GENERATIONS):
        if stats(pop, gen):
            sys.exit()
        pop = step(pop)
    stats(pop, GENERATIONS)

# parameters
GENERATIONS = 100
POPSIZE = 100
MINLEN = 10
MAXLEN = 100
CROSSOVER_PROB = 0.9
PMUT = 0.01
SELECTION = tournament
MAXV = sys.maxint
TOURNAMENT_SIZE = 5
ELITE = 1
WRAPS = 0

# problem-specific
def generate(random):
    return [random.randint(0, 10) for i in range(10)]
def fitness(x):
    return sum(x)
def success(x):
    return x > 90
GENERATE = generate
FITNESS = fitness
SUCCESS = success
MAXIMISE = True

if __name__ == "__main__":
    main()
