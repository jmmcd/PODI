#!/usr/bin/env python

import multiprocessing
import numpy as np
import os, sys
from datetime import datetime
from pylab import *
import scipy
import scipy.stats

from gp import hillclimb
from itertools import product

# import sys
# sys.setrecursionlimit(10000)

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

def hill_climbing_exps():
    hillclimb("vladislavleva-12", "GP", 5, 1, 1, 3)
    hillclimb("vladislavleva-12", "GSGP", 5, 1, 1, 3)
    hillclimb("vladislavleva-12", "GSGP-optimal-ms", 5, 1, 1, 3)


def run1(fn, args, rep, to_file=True):
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
                
def LBYL_experiment():
    try:
        os.makedirs("LBYL")
    except:
        pass
    reps = 10
    fitness_fns = ["vladislavleva-12", "vladislavleva-14", "nguyen-7", "dow-chemical-tower", "evocompetitions-2010", "vanneschi-bioavailability", "pagie-2d"]
    # fitness_fns = ["vladislavleva-14"]
    mut_types = ["GP", "GSGP", "GSGP-optimal-ms"]
    # eval_budget = 1000
    # fitness_fns = ["pagie-2d", "vladislavleva-14", "nguyen-7", "dow-chemical-tower", "evocompetitions-2010", "vanneschi-bioavailability"]
    fitness_fns = ["evocompetitions-2010"]
    fitness_fns = ["pagie-2d"]
    fitness_fns = ["nguyen-7"]
    fitness_fns = ["vladislavleva-12"]
    fitness_fns = ["vladislavleva-14"]
    fitness_fns = ["dow-chemical-tower"]
    fitness_fns = ["vanneschi-bioavailability"]
    mut_types = ["GSGP-optimal-ms", "GP", "GSGP"]
    _eval_budget = 1000
    # ngenss = [1000]
    popsizes = [10]
    # print_everys = [1, 10, 100]
    # st_maxdepthss = [3]

    for rep in range(reps):
        for fitness_fn in fitness_fns:
            for mut_type in mut_types:
                if mut_type == "GP":
                    eval_budget = _eval_budget
                elif mut_type == "GSGP":
                    eval_budget = _eval_budget
                elif mut_type == "GSGP-optimal-ms":
                    eval_budget = _eval_budget / 10
                for popsize in popsizes:
                    ngens = eval_budget / popsize
                    print_every = 10
                    st_maxdepth = 3
                    setup = (fitness_fn, mut_type, ngens, popsize, print_every, st_maxdepth)
                    run1(hillclimb, setup, rep, to_file=True)

if __name__ == "__main__":
    #LBYL_experiment()
    process_hillclimb_dir("LBYL")
    
