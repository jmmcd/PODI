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

from gp import hillclimb, make_fn
import fitness

import sys
sys.setrecursionlimit(100000)

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

def read_dir(dirname):

    fitness_fns = ["GOLD5m", "GOLD1h", "GU5m", "GU1h", "SP5005m", "SP5001h"]
    mutation_types = ["GP", "GSGP", "GSGP-one-tree", "GSGP-optimal-ms"]
    eval_budget = 4000
    st_maxdepths = [3]
    popsizes = [100]
    init_popsizes = [1, "large"]
    print_every = 1
    reps = 10

    p = product(fitness_fns, mutation_types, st_maxdepths, 
                popsizes, init_popsizes)

    returnss = {}

    raw_returns = read_raw_returns()
    
    for item in p:
        (fitness_fn, mutation_type, st_maxdepth, 
         popsize, init_popsize) = item
        if init_popsize == "large":
            init_popsize = popsize
        ngens = round((eval_budget - init_popsize) / float(popsize))
        ngens = eval_budget / popsize
        key = "_".join(map(str, [fitness_fn, mutation_type, st_maxdepth,
                                 popsize, init_popsize]))

        outfilename = dirname + "/" + key
        fig = figure(figsize=(4, 3))
        ax = fig.add_subplot(111)
        ax.set_xlabel("Evaluations")
        ax.set_ylabel("RMSE (Training)")
        # ax.set_title(key)
        fig2 = figure(figsize=(4, 3))
        ax2 = fig2.add_subplot(111)
        ax2.set_xlabel("Time-steps")
        ax2.set_ylabel("Accumulated Returns")
        fig3 = figure(figsize=(4, 3))
        ax3 = fig3.add_subplot(111)
        ax3.set_xlabel("Time-steps")
        ax3.set_ylabel("Accumulated Returns")

        returns_to_plot = []
        for rep in range(reps):
            filename = dirname + "/" + key + "_" + str(rep) + "_numbers.dat"
            d = np.genfromtxt(filename).T
        
            # read in the phenotype separately
            last_tree = open(dirname + "/" + key + "_" + str(rep) + "_trees.dat"
                             ).read().strip("\n").split("\n")[-1]
            returns, sig = get_accumulated_returns(last_tree,
                                                   fitness_fn,
                                                   raw_returns)

            to_print = (fitness_fn, mutation_type, st_maxdepth, 
                        popsize, init_popsize,
                        d[2][-1],
                        returns[50] > 0,
                        returns[100] > 0,
                        returns[-1] > 0,
                        sig)
            print("result %s %s %d %d %d %f %d %d %d %d" % to_print)

            returns_to_plot.append((d[2][-1], returns, sig))

            # plot in grey with transparency
            ax.plot(d[1], d[2],
                    linewidth=3.0, color=(0.3, 0.3, 0.3, 0.6))


        # pick 3 best on training data (RMSE) to run on test data
        returns_to_plot.sort(key=lambda x: x[0])
        for fitness, returns, sig in returns_to_plot[:]:
            ax2.plot(range(len(returns)), returns,
                     linewidth=3.0, color=(0.3, 0.3, 0.3, 0.6))
            ax3.plot(range(50), returns[:50],
                     linewidth=3.0, color=(0.3, 0.3, 0.3, 0.6))

        fig.savefig(outfilename + "_generations.pdf")
        fig2.savefig(outfilename + "_returns.pdf")
        fig3.savefig(outfilename + "_returns_timesteps_50.pdf")
        
        fig.clf()
        fig2.clf()
        fig3.clf()

    #print returnss
        
    
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

def read_raw_returns():
    fitness_fns = ["GOLD5m", "GOLD1h", "GU5m", "GU1h", "SP5005m", "SP5001h"]
    d = {}
    for key in fitness_fns:
        r_t = np.genfromtxt("../data/finance/" + key + ".txt").T[0]
        d[key] = r_t
    return d

def get_accumulated_returns(ind_s, key, raw_returns_d):
    """Given an individual and fitness function, calculate its
    accumulated return."""
    ind = make_fn(fitness.eval_or_exec(ind_s))
    srff = fitness.SymbolicRegressionFitnessFunction.init_from_data_file(
        "../data/finance/" + key + "_gsgp.dat", split=0.7, defn="rmse")
    rmse_fit, yhat = srff.get_semantics(ind, test=True)
    if yhat is None:
        print "yhat is None"
        print ind_s
        return np.zeros_like(srff.test_y), False
    raw_returns = raw_returns_d[key][-len(yhat):]
    # sign(yhat) says whether we buy or short, raw_returns is the true outcome
    our_returns = np.sign(yhat) * raw_returns
    accum_returns = np.add.accumulate(our_returns)
    print len(accum_returns)
    sig = chi2test(raw_returns, yhat)
    return accum_returns, sig

def chi2test(y, yhat):
    signy = np.sign(y)
    signyhat = np.sign(yhat)
    m = np.zeros((3, 3))
    m[1, 1] = np.sum(np.logical_and(signy > 0, signyhat > 0))
    m[1, 2] = np.sum(np.logical_and(signy > 0, signyhat <= 0))
    m[2, 1] = np.sum(np.logical_and(signy <= 0, signyhat > 0))
    m[2, 2] = np.sum(np.logical_and(signy <= 0, signyhat <= 0))
    m[0, 1] = m[1, 1] + m[2, 1]
    m[0, 2] = m[1, 2] + m[2, 2]
    m[1, 0] = m[1, 1] + m[1, 2]
    m[2, 0] = m[2, 1] + m[2, 2]
    m[0, 0] = m[1, 0] + m[2, 0]
    print m
    assert m[0, 0] == m[0, 1] + m[0, 2]
    try:
        x = sum(
            sum(
                ((m[i, j] - m[i, 0] * m[0, j] / m[0, 0]) ** 2.0)
                /
                (m[i, 0] * m[0, j] / m[0, 0])
                for j in range(1, 3))
            for i in range(1, 3)
            )
    except FloatingPointError:
        # can happen eg if rule predicts "up" for every data point
        return False
    # This magic number is the threshold value for chi^2 distribution
    # of 1 degree of freedom at 5% significance level
    if x > 3.8415:
        return True
    else:
        return False

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
    reps = range(10)
    # fitness_fns = ["vladislavleva-12", "vladislavleva-14",
                     # "nguyen-7", "dow-chemical-tower",
                     # "evocompetitions-2010",
                     # "vanneschi-bioavailability", "pagie-2d"]
    fitness_fns = ["GOLD5m", "GOLD1h", "GU5m", "GU1h", "SP5005m", "SP5001h"]
    mutation_types = ["GP", "GSGP", "GSGP-one-tree", "GSGP-optimal-ms"]
    eval_budget = 40000
    st_maxdepths = [2]
    popsizes = [500]
    init_popsizes = [1, "large"]
    print_every = 1

    if not run:
        fitness_vals = np.zeros((len(fitness_fns),
                                 len(mutation_types),
                                 len(st_maxdepths),
                                 len(popsizes),
                                 len(init_popsizes),
                                 len(reps)))

        final_results = np.zeros((len(fitness_fns),
                                  len(mutation_types),
                                  len(st_maxdepths),
                                  len(popsizes),
                                  len(init_popsizes),
                                  len(reps)))

    p = product(fitness_fns, mutation_types, st_maxdepths, 
                popsizes, init_popsizes, reps)

    for item in p:
        (fitness_fn, mutation_type, st_maxdepth, 
         popsize, init_popsize, rep) = item
        if init_popsize == "large":
            init_popsize = popsize
        filename = "/Users/jmmcd/Documents/results/GSGP_finance/" + "_".join(map(str, item)) + ".dat"
        ngens = round((eval_budget - init_popsize) / float(popsize))
        print (fitness_fn, mutation_type, st_maxdepth, ngens,
               popsize, init_popsize, print_every)

        if run:
            cmd = "python gp.py %s %s %d %d %d %d %d " % (
                fitness_fn, mutation_type, st_maxdepth,
                ngens, popsize, init_popsize, print_every)
            print cmd
            redirect = " > " + filename
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
                   st_maxdepths.index(st_maxdepth),
                   popsizes.index(popsize),
                   init_popsizes.index(init_popsize),
                   rep)
            results[idx] = r

    if not run:
        print results

def split_data_files(dirname):
    for file in glob.glob(dirname + "/*.dat"):
        if "_numbers" in file or "_trees" in file:
            # already done
            continue
        cmd = "cut -f1 -d':' " + file + " > " + file[:-4] + "_numbers.dat"
        print cmd
        os.system(cmd)
        cmd = "cut -f2 -d':' " + file + " > " + file[:-4] + "_trees.dat"
        print cmd
        os.system(cmd)

if __name__ == "__main__":
    dirname = "/Users/jmmcd/Dropbox/GSGP-ideas-papers/finance/budget_4000/"
    # split_data_files(dirname)
    read_dir(dirname)
    # LBYL_experiment(run=True)
    # s = "['*', ['+', -0.1, ['+', ['+', ['/', ['*', ['+', ['sin', 'x0'], ['square', 'x0']], ['+', 1.0, ['sqrt', 1.0]]], ['+', -1.0, 'x6']], ['+', ['+', 'x7', 'x3'], ['-', 'x6', 'x0']]], ['sqrt', ['square', 'x5']]]], ['+', ['square', ['/', 'x0', -0.1]], ['*', 'x0', 1.0]]]"
    # k = "GOLD5m"
    # print get_accumulated_returns(s, k)
    
