#!/usr/bin/env python

import multiprocessing
import numpy as np
import os, sys, glob
from datetime import datetime
from itertools import product
from pylab import figure
import scipy
import scipy.stats
import pandas as pd
import matplotlib as mpl

from gp import hillclimb, make_fn
import fitness

import sys
sys.setrecursionlimit(100000)

mpl.rcParams.update({'font.size': 12})

class stardict(dict):
    def getstar(self, *key):
        assert None in key
        # return a list of results
        return [self.__getitem__(k)
                for k in self.keys()
                if all([(k1 == k2 or k1 is None)
                        for k1, k2 in zip(key, k)])]

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


def process_hillclimb_dir(dirname):
    fitness_fns = ["vladislavleva-12", "nguyen-7", "vanneschi-bioavailability", "pagie-2d"]
    mut_types = ["GP", "GSGP", "GSGP-optimal-ms"]

    reps = 10

    for fn in fitness_fns:
        if fn != "vanneschi-bioavailability":
            continue
        for mut_type in mut_types:
            if mut_type == "GSGP-optimal-ms":
                eval_budget = 100
            else:
                eval_budget = 1000
            popsize = 10
            print_every = 10
            st_maxdepth = 3
            ngens = eval_budget / popsize



            out_fname = "_".join(map(str, ["output", fn, mut_type, ngens, popsize, print_every, st_maxdepth])) + ".pdf"
            label = fn + ": " + mut_type

            fig = figure(figsize=(6, 4))
            ax = fig.add_subplot(111)
            ax.set_xlabel("Evaluations")
            ax.set_ylabel("Training Fitness")
            ax.set_title(label)

            devals = []
            dfitness = []
            dtestfitness = []
            for rep in range(reps):

                fname = "_".join(map(str, ["output", fn, mut_type, ngens, popsize, print_every, st_maxdepth, "rep-"+str(rep)]))
                print fname
                with open("LBYL/" + fname) as openf:
                    fevals = []
                    ffitness = []
                    ftestfitness = []
                    for line in openf.read().split("\n"):
                        if len(line) > 1 and (not line.startswith("#")):
                            numbers, phenotype = line.split(" : ", 1)
                            gen, evals, ft, test_ft, length = map(float, numbers.split())
                            fevals.append(evals)
                            ffitness.append(ft)
                            ftestfitness.append(test_ft)
                    devals.append(fevals)
                    dfitness.append(ffitness)
                    dtestfitness.append(ftestfitness)
            mfitness = np.median(dfitness, 0)
            mtestfitness = np.median(dtestfitness)
            ax.plot(devals[0], mfitness, linewidth=3.0)
            fig.savefig("LBYL/" + out_fname)

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


def run1(fn, args, rep, to_file=True):
    """Unused for now."""
    s = "_".join(str(arg) for arg in args) + "_rep-" + str(rep)
    print(s)
    if to_file:
        save = sys.stdout
        sys.stdout = open("LBYL/output_" + s, "w")
    n1 = datetime.datetime.now()
    fn(*args)
    n2 = datetime.datetime.now()
    if to_file:
        sys.stdout = save
    print("elapsed time %s\n" % str(n2 - n1))

def LBYL_experiment(run=True):
    reps = range(30)
    # fitness_fns = ["vladislavleva-12", "vladislavleva-14",
                     # "nguyen-7", "dow-chemical-tower",
                     # "evocompetitions-2010",
                     # "vanneschi-bioavailability", "pagie-2d"]
    fitness_fns = ["GOLD5m", "GU5m", "SP5005m"]
    mutation_types = ["GP", "GSGP", "GSGP-one-tree", "GSGP-optimal-ms"]
    # mutation_types = ["GSGP-one-tree", "GSGP-optimal-ms"]
    eval_budget = 20000
    rt_methods = ["grow"]
    rt_sizes = [2]
    popsizes = [500]
    init_popsizes = ["large"]
    print_every = 1

    p = product(reps, fitness_fns, mutation_types,
                rt_methods, rt_sizes,
                popsizes, init_popsizes)

    for item in p:
        (rep, fitness_fn, mutation_type, rt_method, rt_size,
         popsize, init_popsize) = item
        if init_popsize == "large":
            init_popsize = popsize
        filename = "/Users/jmmcd/Dropbox/GSGP-ideas-papers/finance/budget_20000_nlags_20/" + "_".join(map(str, item)) + ".dat"
        ngens = round((eval_budget - init_popsize) / float(popsize))
        # print (fitness_fn, mutation_type,
        #        rt_method, rt_size,
        #        ngens,
        #        popsize, init_popsize, print_every)
        rt_method = "grow"

        if run:
            cmd = "python gp.py %s %s %s %d %d %d %d %d " % (
                fitness_fn, mutation_type, rt_method, rt_size,
                ngens, popsize, init_popsize, print_every)
            redirect = " > " + filename
            print cmd + redirect
            # run this as a system command, redirecting output to a file,
            # because it avoids danger of leaking memory if everything is
            # inside a single big process
            os.system(cmd + redirect)
        else:
            # process the result: get the outcomes, put them into a 6D
            # matrix

            numbers, phenotype = open(filename).readlines()[-1].split(":")
            numbers = map(float, numbers.split())
            print phenotype

            if "GOLD" in fitness_fn or "GU" in fitness_fn or "SP500" in fitness_fn:
                # accumulated returns on test data
                r = get_accumulated_returns(phenotype, fitness_fn)
            else:
                # MSE or whatever on test data
                r = numbers[3]

            idx = (fitness_fns.index(fitness_fn),
                   mutation_types.index(mutation_type),
                   rt_sizes.index(rt_size),
                   popsizes.index(popsize),
                   init_popsizes.index(init_popsize),
                   rep)
            results[idx] = r

    if not run:
        print results

def split_data_files(dirname):
    for file in glob.glob(dirname + "/*.dat"):
        if "_numbers.dat" in file or "_trees.dat" in file or "_returns.dat" in file or "_gens.dat" in file:
            # already done
            continue

        # assume 39 lines of generations and 418 of returns
        headout = file[:-4] + "_gens.dat"
        head = "head -39 " + file + " > " + headout
        tailout = file + " > " + file[:-4] + "_returns.dat"
        tail = "tail -418 " + file + " > " + tailout
        cut1out = file[:-4] + "_numbers.dat"
        cut1 = "cut -f1 -d':' " + headout + " > " + cut1out
        cut2out = file[:-4] + "_trees.dat"
        cut2 = "cut -f2 -d':' " + headout + " > " + cut2out

        print head
        print tail
        print cut1
        print cut2
        # os.system(head)
        # os.system(tail)
        # os.system(cut1)
        # os.system(cut2)

if __name__ == "__main__":
    dirname = "/Users/jmmcd/Dropbox/GSGP-ideas-papers/finance/budget_20000_nlags_20/"
    split_data_files(dirname)
    # LBYL_experiment(run=True)
    # s = "['*', ['+', -0.1, ['+', ['+', ['/', ['*', ['+', ['sin', 'x0'], ['square', 'x0']], ['+', 1.0, ['sqrt', 1.0]]], ['+', -1.0, 'x6']], ['+', ['+', 'x7', 'x3'], ['-', 'x6', 'x0']]], ['sqrt', ['square', 'x5']]]], ['+', ['square', ['/', 'x0', -0.1]], ['*', 'x0', 1.0]]]"
    # k = "GOLD5m"
    # print get_accumulated_returns(s, k)
