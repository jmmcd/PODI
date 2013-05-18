#!/usr/bin/env python

import multiprocessing
import numpy as np
import os, sys
from pylab import *
import scipy
import scipy.stats

from gp import hillclimb, srff
from itertools import product

# TODO use this style instead
# date, rain, high, low = zip(*csv.reader(file("weather.csv")))

rep_titles = {"grow": "Grow", "bubble_down": "Bubble-down"}
dist_titles = {
    "genotype": "Genotype",
    "phenotype": "Phenotype",
    "semantics": "Semantic",
    "fitness": "Fitness"
    }

def gp_distances_boxplots(basedir):
    print(r"\begin{tabular}{l|l|l|l|l}")
    print(r" Representation & Distance & Random & Mutation & Crossover \\ \hline")
    
    reps = ["grow", "bubble_down"]
    for rep in reps:
        ops = ["random", "mutation", "crossover"]
        process(basedir, ops, rep)
        
    print(r"\end{tabular}")        

def process(basedir, ops, rep):
    data = {}
    dtypes = ["genotype", "phenotype", "semantics", "fitness"]
    for op in ops:
        filename = os.path.join(basedir, rep, op + "_distances.dat")
        txt = open(filename).read()
        _data = []
        for line in txt.split("\n"):
            if len(line) < 2: continue
            parts = line.split()
            g, p, s, f = parts
            g = int(g)
            p = int(p)
            s = float(s)
            f = float(f)
            if g == 0: continue
            _data.append((g, p, s, f))
        _data = np.array(_data)
        _data = _data.transpose()
        data[op] = _data
    
    for i, d in enumerate(dtypes):
        bpdata = [data[op][i] for op in ops]
        # make_boxplot(basedir, ops, rep, d, bpdata)
        print_data(rep, d, bpdata)
        
def make_boxplot(basedir, ops, rep, dist, bpdata):
    codename = rep + "_" + dist
    fig = figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.boxplot(bpdata)
    if rep in ["semantics", "fitness"]:
        ax.set_yscale('log')
    ax.set_xticklabels(ops)
    ax.set_ylabel(dist + " distance")
    outfilename = os.path.join(basedir, codename + ".pdf")
    print("saving to " + outfilename)
    fig.savefig(outfilename)
    close()

def print_data(rep, dist, bpdata):
    # we use Mann-Whitney non-parametric test
    # because fitness values have many, huge outliers
    
    # random v mutation
    mu, mp = scipy.stats.mannwhitneyu(bpdata[0], bpdata[1])
    # multiply by 2 for two-sided 
    mp *= 2.0
    # random v crossover
    cu, cp = scipy.stats.mannwhitneyu(bpdata[0], bpdata[2])
    # multiply by 2 for two-sided 
    cp *= 2.0

    if mp < 0.01:
        mps = r"{\bf *}"
    else:
        mps = ""
    if cp < 0.01:
        cps = r"{\bf *}"
    else:
        cps = ""
    
    rmean = np.mean(bpdata[0])
    rmedian = np.median(bpdata[0])
    rstddev = np.std(bpdata[0])
    mmean = np.mean(bpdata[1])
    mmedian = np.median(bpdata[1])
    mstddev = np.std(bpdata[1])
    cmean = np.mean(bpdata[2])
    cmedian = np.median(bpdata[2])
    cstddev = np.std(bpdata[2])

    print(rep)
    print(dist)
    print("random v mutation", mu, mp)
    print("random v crossover", cu, cp)
    print("random", rmean, rstddev, rmedian)
    print("mutation", mmean, mstddev, mmedian)
    print("crossover", cmean, cstddev, cmedian)

    print(r" %s & %s & %.2g & %.2g \hfill %s & %.2g \hfill %s \\" % (
            rep_titles[rep],
            dist_titles[dist],
            rmedian, 
            mmedian, mps,
            cmedian, cps))

    # print(r"\begin{tabular}{l|ll|lll|lll}")
    # print(r"         & Random      &      & Mutation       &       & Crossover       &       \\")
    # print(r"Distance & mean (sd)   & med  & mean (sd)      & med   & mean (sd)       & med   \\ \hline")
    # print(r" %s      & %.1g (%.1g) & %.1g & %.1g (%.1g) %s & %.1g  & %.1g (%.1g) %s  & %.1g  \\" % (
    #         dist,
    #         rmean, rstddev, rmedian, 
    #         mmean, mstddev, mps, mmedian,
    #         cmean, cstddev, cps, cmedian))
    # print(r"\end{tabular}")

def hill_climbing_exps():
    hillclimb(srff, "GP", 5, 1, 1, 3)
    hillclimb(srff, "GSGP", 5, 1, 1, 3)
    hillclimb(srff, "GSGP-optimal-ms", 5, 1, 1, 3)


def run1(fn, args, rep, to_file=True):
    s = "-".join(str(arg) for arg in args) + "_rep-" + str(rep)
    print(s)
    if to_file:
        save = sys.stdout
        sys.stdout = open("LBYL/output_" + s, "w")
    fn(*args)
    if to_file:
        sys.stdout = save
                
def LBYL_experiment():
    try:
        os.makedirs("LBYL")
    except:
        pass
    reps = 2
    fitness_fns = [srff]
    mut_types = ["GP", "GSGP", "GSGP-optimal-ms"]
    ngenss = [5]
    popsizes = [1]
    print_everys = [1]
    st_maxdepthss = [2, 3]
    for rep in range(reps):
        print("rep ", rep)
        p = product(fitness_fns, mut_types, ngenss,
                    popsizes, print_everys, st_maxdepthss)
        for setup in p:
            run1(hillclimb, setup, rep, to_file=True)

if __name__ == "__main__":
    LBYL_experiment()
