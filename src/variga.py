#!/usr/bin/env python

import sys, time, random
import numpy as np
import nonrandom
from copy import deepcopy
from collections import namedtuple

fitness_idx, used_idx, genome_idx, phenotype_idx, readable_phenotype_idx, semantics_idx = range(6)
    
# If you try to compare individuals directly, using tuple comparison,
# you can get in trouble when the genomes have different lengths and
# the fitness and used values are the same, because max() will compare
# the next element of the tuples, the genomes, and max() on numpy
# arrays will try to produce a new array of the pointwise max
# values. So we only use the fitness. High fitness is good for
# MAXIMISE, else low fitness.
def ind_compare(x):
    if MAXIMISE:
        return x.fitness
    else:
        return -x.fitness

Individual = namedtuple("Individual", ["genome", "used_codons", "fitness",
                                       "phenotype", "readable_phenotype", "semantics"])

def make_individual(genome):
    if genome is None:
        LEN = random.randint(MINLEN, MAXLEN)
        genome = [random.randint(0, MAXV-1) for i in range(LEN)]
    nr = nonrandom.NonRandom(genome, maxval=MAXV, wraps=WRAPS)
    try:
        readable_phenotype, phenotype = GENERATE(nr)
        fitness, semantics = FITNESS.get_semantics(phenotype)
    except StopIteration:
        phenotype = None
        readable_phenotype = None
        semantics = None
        if MAXIMISE:
            fitness = -float("inf")
        else:
            fitness = float("inf")
    used = nr.used
    return Individual(fitness=fitness, used_codons=used, genome=genome,
                      phenotype=phenotype,
                      readable_phenotype=readable_phenotype,
                      semantics=semantics)

# Onepoint crossover FIXME change to twopoint?
def xover(a, b):
    g, h = a.genome, b.genome
    if random.random() < CROSSOVER_PROB:
        # -1 to get last index in array; min() in case of wraps: used > len
        max_g, max_h = min(len(g)-1, a.used_codons), min(len(h)-1, b.used_codons)
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
    valids = [i for i in pop if i.phenotype is not None]
    ninvalids = len(pop) - len(valids)
    if len(valids) == 0:
        fitness_vals = np.array([0])
        used_vals = np.array([0])
    else:
        fitness_vals = np.array([i.fitness for i in valids])
        used_vals = np.array([i.used_codons for i in valids])
    len_vals = np.array([len(i.genome) for i in pop])
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
                  best.fitness, best.used_codons,
                  FITNESS.test(best.phenotype),
                  meanfit, sdfit, meanused, sdused,
                  meanlen, sdlen, ninvalids, 
                  best.readable_phenotype))
    return SUCCESS(best.fitness)

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
    newpop = [make_individual(g) for g in newpop]
    
    # elite: replace worst
    newpop.sort(key=ind_compare)
    newpop[:ELITE] = elite

    if COEVOLUTIONARY_FITNESS:
        COEVOLUTIONARY_FITNESS(newpop)
    return newpop
    
def main(seed=None):
    if seed is not None:
        random.seed(seed)
    pop = [make_individual(None) for i in range(POPSIZE)]
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
    x = [random.randint(0, 10) for i in range(10)]
    return (x, x)
class fitness:
    def __call__(self, x):
        return self.get_semantics(x)[0]
    def test(self, x):
        return self(x)
    def get_semantics(self, x):
        return sum(x), x
def success(x):
    return x > 90
GENERATE = generate
FITNESS = fitness()
COEVOLUTIONARY_FITNESS = None
SUCCESS = success
MAXIMISE = True

if __name__ == "__main__":
    main()
