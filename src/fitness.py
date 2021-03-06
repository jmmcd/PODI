#!/usr/bin/env python

import os, sys
import random
import operator
from itertools import product

from scipy.stats import ks_2samp
import numpy as np
from numpy import add, subtract, multiply, divide, sin, cos, exp, log, power, square
from numpy import logical_and, logical_or, logical_xor, logical_not
np.seterr(all='raise', under='ignore')
from sklearn.linear_model import LinearRegression, ElasticNet
try:
    import editdist
except:
    # StringMatch won't be available. See reqs.txt (TODO or could fall
    # back to the levenshtein function in structure.py in this case.)
    pass

def eval_or_exec(expr):
    """Use eval or exec to interpret expr.

    A limitation in Python is the distinction between eval and
    exec. The former can only be used to return the value of a simple
    expression (not a statement) and the latter does not return
    anything."""

    # print(expr)
    try:
        retval = eval(expr)
    except SyntaxError:
        # SyntaxError will be thrown by eval() if s is compound,
        # ie not a simple expression, eg if it contains function
        # definitions, multiple lines, etc. Then we must use
        # exec(). Then we assume that s will define a variable
        # called "XXXeval_or_exec_outputXXX", and we'll use that.
        dictionary = {}
        exec(expr, dictionary)
        retval = dictionary["XXXeval_or_exec_outputXXX"]
    except MemoryError:
        # Will be thrown by eval(s) or exec(s) if s contains over-deep
        # nesting (see http://bugs.python.org/issue3971). The amount
        # of nesting allowed varies between versions, is quite low in
        # Python2.5. If we can't evaluate, there is an awful hack:
        # write out a file and import from it. This works because the
        # compiler itself is not recursive, whereas eval() is.
        try:
            import imp, tempfile
            f = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py")
            f.write("XXXeval_or_exec_outputXXX = " + expr)
            f.close()
            m = imp.load_source("tmp", f.name)
            retval = m.XXXeval_or_exec_outputXXX
        except MemoryError:
            # But there is still a limit on the depth which can be
            # evaluated. If we hit that limit, then give up.
            retval = None
    return retval

def default_fitness(maximise):
    """Return default (worst) fitness, given maximization or
    minimization."""
    if maximise:
        return -sys.maxint
    else:
        return sys.maxint

class RandomFitness:
    """Useful for investigating algorithm dynamics in the absence of
    selection pressure. Fitness is random."""
    def __call__(self, ind):
        """Allow objects of this type to be called as if they were
        functions. Return a random value as fitness."""
        return self.get_semantics(ind)[0]
    def test(self, ind):
        return self(ind)
    def get_semantics(self, ind):
        x = random.random()
        return (x, x)

class SizeFitness:
    """Useful for investigating control of tree size. Return the
    difference from a target size."""
    maximise = False
    def __init__(self, target_size=20):
        self.target_size = target_size
    def __call__(self, ind):
        """Allow objects of this type to be called as if they were
        functions."""
        return self.get_semantics(ind)[0]
    def test(self, ind):
        return self(ind)
    def get_semantics(self, ind):
        ln = len(ind)
        return abs(self.target_size - ln), ln

class MaxFitness():
    """Arithmetic maximisation with python evaluation."""
    maximise = True
    def __call__(self, ind):
        return self.get_semantics(ind)[0]
    def test(self, ind):
        return self(ind)
    def get_semantics(self, ind):
        x = eval_or_exec(ind)
        return (x, x)

class StringMatch():
    """Fitness function for matching a string. Takes a string and
    returns fitness. Usage: StringMatch("golden") returns a *callable
    object*, ie the fitness function. Uses Levenshtein edit distance
    provided by the editdist library: see reqs.txt."""
    maximise = False
    def __init__(self, target):
        self.target = target
    def __call__(self, ind):
        return self.get_semantics(ind)[0]
    def test(self, ind):
        return self(ind)
    def get_semantics(self, ind):
        return editdist.distance(self.target, ind), ind

class StringProc():
    """Fitness function for programs that process strings, ie they
    have inputs which are supposed to map to outputs."""
    maximise = False
    def __init__(self, test_cases):
        self.test_cases = test_cases
    def __call__(self, p):
        return self.get_semantics(p)[0]
    def test(self, ind):
        # TODO should allow for a proper train/test suite
        return self(ind)
    def get_semantics(self, ind):
        semantics = []
        fitness = 0
        for input, output in self.test_cases:
            result = p(input)
            semantics.append(result)
            fitness += editdist(result, output)
        return fitness, semantics

class BooleanProblem:
    """Boolean problem of size n. Pass target function in.
    Minimises. Objects of this type can be called."""

    # TODO could benchmark this versus sub-machine code
    # implementation.
    def __init__(self, n, target):
        self.maximise = False

        # make all possible fitness cases
        vals = [False, True]
        p = list(product(*[vals for i in range(n)]))
        self.x = np.transpose(p)

        # get target function's values on fitness cases
        try:
            # assume target is a function
            self.target_cases = target(self.x)
        except TypeError:
            # no, target was a list of values
            if len(target) != 2 ** n:
                s = "Wrong number of target cases (%d) for problem size %d" % (
                    len(target), n)
                raise ValueError(s)
            self.target_cases = np.array(target)

    def __call__(self, s):
        return self.get_semantics(s)[0]

    def test(self, ind):
        # TODO should allow for a proper train/test suite
        return self(ind)

    def get_semantics(self, ind):
        # s is a string which evals to a fn.
        fn = eval(s)
        output = fn(self.x)
        non_matches = output ^ self.target_cases
        return output, sum(non_matches) # Fitness is number of errors


class BooleanProblemGeneral:
    """A compound problem. It consists of a single Boolean problem, at
    several sizes. Some sizes are used for training, some for
    testing. Training fitness is the sum of fitness on the training
    sub-problems. Testing fitness is the sum of fitness on the testing
    sub-problems."""
    def __init__(self, train_ns, test_ns, target):
        self.maximise = False
        self.train_problems = [
            BooleanProblem(n, target) for n in train_ns]
        self.test_problems = [
            BooleanProblem(n, target) for n in test_ns]
    def __call__(self, s):
        return self.get_semantics(s)[0]
    def test(self, s):
        return sum([p(s) for p in self.test_problems])
    def get_semantics(self, s):
        semantics = [p.get_semantics(s) for p in self.train_problems]
        # sum(semantics_i[0]) is the sum of fitness values
        # [semantics_i[1]] is a list of numpy arrays.
        return (sum(semantics_i[0] for semantics_i in semantics),
                [semantics_i[1] for semantics_i in semantics])


class ProbabilityDistributionFitnessFunction:
    """Fitness function for running GP to create a model of an unknown
    probability distribution. This problem is a bit like symbolic
    regression, but not exactly. We start off with just a list of
    numbers, which we assume are sampled from some unknown
    distribution, and our goal is to find a function which matches
    that distribution.

    There are several ways we could go about this. For now, this
    implements a fairly simple method: candidate functions can use a
    RAND call to get random numbers. The outputs of the function are
    then tested using the Kolmogorov-Smirnov 2-sample test against the
    known samples.

    Probably the right way to do this is to parse fn to get a PyMC
    model, then fit it, then see how well it does in either a PyMC
    goodness of fit test, or just the ks_2samp as above.

    Another option is to generate fn to take a single argument, which
    is a number between 0 and 1, and regard fn as the generating
    function of the distribution, ie the inverse of the cumulative
    distribution function."""

    def __init__(self, x, n=100):
        # from pymc import *
        # samples from an unknown distribution to be matched
        self.x = x
        # how many samples should we generate when testing candidates?
        self.n = n

    def __call__(self, fn):
        return self.get_semantics(fn)[0]
    def test(self, fn):
        return self(fn) # FIXME do something more useful here
    def get_semantics(self, fn):
        if not callable(fn):
            # assume fn is a string which evals to a function.
            try:
                fn = eval(fn)
            except MemoryError:
                return default_fitness(self.maximise), None

        fn_data = fn(np.zeros((1, self.n)))

        # print("fn_data")
        # print(fn_data)
        # Test against our data. ks_2samp returns (D, p), where D is
        # the KS statistic. if the KS statistic is small or the
        # p-value is high, then we cannot reject the hypothesis that
        # the distributions of the two samples are the same. In other
        # words, smaller D means the distributions are more similar.
        # So we're aiming to minimise D.
        return ks_2samp(self.x, fn_data)[0], fn_data


class SymbolicRegressionFitnessFunction:
    """Fitness function for symbolic regression problems. Yes, it's a
    Verb in the Kingdom of Nouns
    (http://steve-yegge.blogspot.com/2006/03/execution-in-kingdom-of-nouns.html).
    The reason is that the function needs some extra data to go with
    it: an arity, a list of bounds and increments to create the test
    cases. Actually there are lots of bits and pieces and it's best to
    keep them together."""

    def __init__(self, train_X, train_y, test_X=None, test_y=None,
                 defn="rmse"):
        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y
        self.arity = len(train_X)
        if defn == "rmse":
            self.maximise = False
            self.defn = self.rmse
        elif defn == "correlation":
            self.maximise = False
            self.defn = self.correlation_fitness
        elif defn == "log_error":
            self.maximise = False
            self.defn = self.log_error
        elif defn == "hits":
            self.maximise = True
            self.defn = self.hits_fitness
        elif defn == "classification_accuracy":
            self.maximise = True
            self.defn = self.class_acc_fitness
        else:
            raise ValueError("Bad value for fitness definition: " + defn)

    @classmethod
    def init_from_data_file(cls, filename, test_filename=None,
                            split=0.9,
                            randomise=False, defn="rmse"):
        """Construct an SRFF by reading training data from a file. If
        a test_filename is given, get test data there. Else split the
        data according to split, eg 0.9 means 90% for training, 10%
        for testing. If randomise is True, take random rows; else take
        the last rows as test data. Tries to handle csv files
        correctly, else genfromtxt assumes it is whitespace-separated.
        # character is used for comment lines."""

        if filename.endswith(".csv"):
            delimiter = ","
        else:
            delimiter = None
        d = np.genfromtxt(filename, delimiter=delimiter)
        if randomise:
            # shuffle the rows before allocating train/test
            np.random.shuffle(d)
        dX = d[:,:-1]
        dy = d[:,-1]
        if test_filename:
            # separate filename for test
            train_X = dX.T
            train_y = dy
            # assume same filename suffix and delimiter
            d = np.genfromtxt(test_filename, delimiter=delimiter)
            test_X = d[:,:-1].T
            test_y = d[:,-1]
        elif split:
            # get data from same file for both train and test
            idx = int(split * len(d))
            train_X = dX[:idx].T
            train_y = dy[:idx]
            test_X = dX[idx:].T
            test_y = dy[idx:]
        else:
            # same data for train and test
            test_X = train_X = dX.T
            test_y = train_y = dy

        # print "len(train_X), len(test_X)"
        # print len(train_X), len(test_X)
        # print "len(train_X[0]), len(train_y), len(test_X[0]), len(test_y)"
        # print len(train_X[0]), len(train_y), len(test_X[0]), len(test_y)
        assert len(train_X.shape) == len(test_X.shape) == 2
        assert len(train_y.shape) == len(test_y.shape) == 1
        assert len(train_X) == len(test_X) # how many independent vars
        assert len(train_X[0]) == len(train_y)
        assert len(test_X[0]) == len(test_y)
        return SymbolicRegressionFitnessFunction(train_X, train_y,
                                                 test_X, test_y, defn)

    @classmethod
    def init_from_target_fn(cls, target, train, test1=None, test2=None, defn="rmse"):
        """Pass in a target function and parameters for building the
        fitness cases for input variables for training and for testing
        if necessary. The cases are specified as a dictionary
        containing bounds and other variables: either a regular mesh
        over the ranges or randomly-generated points within the ranges
        can be generated. We allow one set of cases for training, and
        zero, one or two for testing, because several of our benchmark
        functions need two discontinuous ranges for testing data. If
        no testing cases are specified, the training data is used for
        testing as well.

        Can pass a keyword indicating the fitness definition, which
        can be 'rmse' (minimise root mean square error), 'correlation'
        (minimise 1-r**2), or 'hits' (maximise number of fitness cases
        within a small threshold)."""

        # Training data
        cases = cls.build_column_mesh_np(cls.build_cases(**train))
        values = target(cases)

        # Testing data -- FIXME this could be neater.
        if test1 and test2:
            testing_cases = cls.build_column_mesh_np(cls.build_cases(**test1) +
                                                           cls.build_cases(**test2))
            testing_values = target(testing_cases)
        elif test1:
            testing_cases = cls.build_column_mesh_np(cls.build_cases(**test1))
            testing_values = target(testing_cases)
        else:
            # No special testing cases -- use training cases
            testing_cases = cases
            testing_values = values
        return SymbolicRegressionFitnessFunction(cases, values,
                                                 testing_cases,
                                                 testing_values, defn)


    def __call__(self, fn):
        """Allow objects of this type to be called as if they were
        functions. Return just a fitness value."""
        return self.get_semantics(fn)[0]

    def get_semantics(self, fn, test=False):
        """Run the function over the training set. Return the fitness
        and the vector of results (the "semantics" of the function).
        Return (default_fitness, None) on error. Pass test=True to run
        the function over the testing set instead."""

        # print("s:", s)
        if not callable(fn):
            # assume fn is a string which evals to a function.
            try:
                fn = eval(fn)
            except MemoryError:
                # print("MemoryError in get_semantics()")
                # self.memo[s, test] = default_fitness(self.maximise), None
                # return self.memo[s, test]
                return default_fitness(self.maximise), None

        try:
            if not test:
                vals_at_cases = fn(self.train_X)
                fit = self.defn(self.train_y, vals_at_cases)
            else:
                vals_at_cases = fn(self.test_X)
                fit = self.defn(self.test_y, vals_at_cases)

            return fit, vals_at_cases
            # self.memo[s, test] = fit, vals_at_cases
            # return self.memo[s, test]

        except FloatingPointError as fpe:
            # print("FloatingPointError in get_semantics()")
            return default_fitness(self.maximise), None
            # self.memo[s, test] = default_fitness(self.maximise), None
            # return self.memo[s, test]
        except ValueError as ve:
            print("ValueError: " + str(ve) +':' + str(fn))
            raise
        except TypeError as te:
            print("TypeError: " + str(te) +':' + str(fn))
            raise

    def test(self, fn):
        """Test ind on unseen data. Return a fitness value."""
        return self.get_semantics(fn, True)[0]

    @staticmethod
    def build_cases(minv, maxv, incrv=None, randomx=None, ncases=None):
        """Generate fitness cases, either randomly or in a mesh."""
        if randomx is True:
            # incrv is ignored
            return SymbolicRegressionFitnessFunction.build_random_cases(minv, maxv, ncases)
        else:
            # ncases is ignored
            return SymbolicRegressionFitnessFunction.build_mesh(minv, maxv, incrv)

    @staticmethod
    def rmse(y, yhat):
        """Calculate root mean square error between yhat and y, two numpy
        arrays."""
        m = yhat - y
        m = np.square(m)
        m = np.mean(m)
        m = np.sqrt(m)
        return m

    @staticmethod
    def log_error(y, yhat):
        """Calculate log error between yhat and y, two numpy arrays."""
        return 1.0 * np.mean(np.log(1.0+(np.abs(yhat-y))))

    @staticmethod
    def hits_fitness(y, yhat):
        """Hits as fitness: how many of the errors are very small?
        Minimise 1 - the proportion."""
        errors = abs(yhat - y)
        return 1 - np.mean(errors < 0.01)

    @staticmethod
    def class_acc_fitness(y, yhat):
        """Classification accuracy fitness: how many of the
        predictions are of the same sign as the correct value?
        Maximise."""
        signy = np.sign(y)
        signyhat = np.sign(yhat)
        tp_plus_tn = np.count_nonzero((signy * signyhat) >= 0)
        return tp_plus_tn / float(len(y))

    @staticmethod
    def correlation_fitness(y, yhat):
        """Correlation coefficient as fitness: minimise 1 - R^2."""
        try:
            # use [0][1] to get the right element from corr matrix
            corr = abs(np.corrcoef(yhat, y)[0][1])
        except (ValueError, FloatingPointError):
            # ValueError raised when yhat is a scalar because individual does
            # not depend on input. FloatingPointError raised when elements
            # of yhat are all identical.
            corr = 0.0
        return 1.0 - corr * corr

    def predict_constant(self):
        """How well can we do on this SRFF by predicting a
        constant?"""
        mn = np.mean(self.train_y)
        # generate a constant column
        yhat = np.ones_like(self.test_y) * mn
        err = self.defn(self.test_y, yhat)
        return mn, err

    def predict_linear(self, enet=True):
        """How well can we do on this SRFF with a linear regression
        (with optional elastic-net regularisation)?"""
        if enet:
            clf = ElasticNet()
        else:
            clf = LinearRegression()
        # we have to transpose X here because sklearn uses the
        # opposite order (rows v columns). maybe this is a sign that
        # I'm using the wrong order.
        clf.fit(self.train_X.T, self.train_y)
        yhat = clf.predict(self.test_X.T)
        err = self.defn(self.test_y, yhat)
        return clf.intercept_, clf.coef_, err

    @staticmethod
    def build_random_cases(minv, maxv, n):
        """Create a list of n lists, each list being x-coordinates for a
        fitness case. Generate them randomly within the bounds given by
        minv and maxv."""
        return [[random.uniform(lb, ub) for lb, ub in zip(minv, maxv)]
                for i in range(n)]

    @staticmethod
    def build_mesh(minv, maxv, increment):
        """Build a mesh, i.e. enumerate all points within the volume
        specified by the minv and maxv lists, at increment distances
        apart. Uses itertools.product.

        Two constraints on the input parameters: The three lists provided
        as parameters should be of the same length; and minv[i] <= maxv[i]
        for all i.

        Thanks to David White for an original Java implementation of this
        code, and comments.

        FIXME this should be doable using numpy meshgrid methods.

        @param minv A list of minimum values for the n variables.
        @param maxv A list of maximum values for the n variables.
        @param increment A list of increments for the n variables.
        @return A list of tuples. Each tuple is the coordinates of a
        particular point."""

        assert len(minv) == len(maxv) == len(increment)
        one_d_meshes = []
        for minvi, maxvi, inci in zip(minv, maxv, increment):
            assert minvi <= maxvi
            nsteps = int((maxvi - minvi) / float(inci)) # eg [0, 10, 1] gives 10
            # note +1 to reach max
            mesh = [minvi + inci * i for i in range(nsteps + 1)]
            one_d_meshes.append(mesh)
        p = list(product(*one_d_meshes))
        return p

    @staticmethod
    def test_build_random():
        """Test -- this should print 100 points in correct ranges."""
        minv = [0.0, 0.0]
        maxv = [2.0, 2.0]
        n = 100
        mesh = SymbolicRegressionFitnessFunction.build_random_cases(minv, maxv, n)
        print(len(mesh))
        for item in mesh:
            print(item)

    @staticmethod
    def test_build_mesh():
        """Test -- this should print 63 points:
        [0.0, 0.0]
        [0.0, 1.0]
        [0.0, 2.0]
        [...]
        [2.0000000000000004, 0.0]
        [2.0000000000000004, 1.0]
        [2.0000000000000004, 2.0]"""
        minv = [0.0, 0.0]
        maxv = [2.0, 2.0]
        incrv = [0.1, 1.0]
        mesh = SymbolicRegressionFitnessFunction.build_mesh(minv, maxv, incrv)
        print(len(mesh))
        for item in mesh:
            print(item)

    @staticmethod
    def build_column_mesh_np(in_mesh):
        """Given a mesh of fitness cases, build a column-wise mesh
        consisting of multiple numpy arrays. Each array represents the
        values of an input variable."""
        mesh = np.array(in_mesh)
        mesh = mesh.transpose()
        return mesh

def benchmarks(key):
    if   key == "identity":
        return SymbolicRegressionFitnessFunction.init_from_target_fn(
            lambda x: x,
            {"minv": [0.0], "maxv": [1.0], "incrv": [0.1]})

    elif key == "vladislavleva-12":
        return SymbolicRegressionFitnessFunction.init_from_target_fn(
            lambda x: exp(-x[0]) * power(x[0], 3.0) * cos(x[0]) * sin(x[0]) \
                * ((cos(x[0]) * power(sin(x[0]), 2.0)) - 1.0),
            {"minv": [0.05], "maxv": [10.0], "incrv": [0.1]},
            test1={"minv": [-0.5], "maxv": [10.5], "incrv": [0.05]})

    elif key == "vladislavleva-14":
        return SymbolicRegressionFitnessFunction.init_from_target_fn(
            lambda x: 10.0 / (5.0 + sum((x[i]-3)**2 for i in range(5))),
            {"minv": [0.05]*5, "maxv": [6.05]*5, "randomx": True, "ncases": 1024},
            test1={"minv": [-0.25]*5, "maxv": [6.35]*5, "randomx": True, "ncases": 5000})

    elif key == "korns-12":
        # Note need to use bnf with x[0] -- x[4]
        # FIXME the number of variables in the grammar should be inferred
        # from the data
        return SymbolicRegressionFitnessFunction.init_from_target_fn(
            lambda x: 2.0 - (2.1 * (cos (9.8*x[0])*sin(1.3*x[1]))),
            {"minv": [-50,-50,-50,-50,-50],
             "maxv": [50,50,50,50,50], "randomx": True, "ncases": 10000},
            test1={"minv": [-50,-50,-50,-50,-50],
                   "maxv": [50,50,50,50,50,50], "randomx": True, "ncases": 10000})

    elif key == "pagie-2d":
        return SymbolicRegressionFitnessFunction.init_from_target_fn(
            lambda x: (1 / (1 + x[0] ** -4) + 1 / (1 + x[1] ** -4)),
            {"minv": [-5, -5], "maxv": [5, 5], "incrv": [0.4, 0.4]})

    elif key == "pagie-3d":
        return SymbolicRegressionFitnessFunction.init_from_target_fn(
            lambda x: (1 / (1 + x[0] ** -4) + 1 / (1 + x[1] ** -4)
                       + 1 / (1 + x[2] ** -4)),
            {"minv": [-5, -5, -5], "maxv": [5, 5, 5], "incrv": [0.4, 0.4, 0.4]})

    elif key == "vanneschi-bioavailability":
        return SymbolicRegressionFitnessFunction.init_from_data_file(
            "../data/bioavailability.txt", split=0.7, randomise=True)

    elif key == "vanneschi-protein":
        return SymbolicRegressionFitnessFunction.init_from_data_file(
            "../data/ppb.txt", split=0.7, randomise=True)

    elif key == "vanneschi-toxicity":
        return SymbolicRegressionFitnessFunction.init_from_data_file(
            "../data/toxicity.txt", split=0.7, randomise=True)

    elif key == "dow-chemical-tower":
        return SymbolicRegressionFitnessFunction.init_from_data_file(
            "../data/towerData.txt", split=0.7, randomise=True)

    elif key == "evocompetitions-2010":
        return SymbolicRegressionFitnessFunction.init_from_data_file(
            "../data/evocompetitions_2010/evocompetitions_2010_train.csv",
            test_filename="../data/evocompetitions_2010/evocompetitions_2010_test.csv")

    elif key == "nguyen-7":
        return SymbolicRegressionFitnessFunction.init_from_target_fn(
            lambda x: log(x[0] + 1) + log(x[0]**2 + 1),
            {"minv": [0], "maxv": [2], "randomx": True, "ncases": 20})

    elif key == "fagan":
        return SymbolicRegressionFitnessFunction.init_from_target_fn(
            lambda x: x[0]**4 + x[1]**2,
            {"minv": [-5, -5], "maxv": [5, 5], "incrv": [0.4, 0.4]},
            defn="log_error")

    elif key in ["GOLD1h", "GOLD5m", "GU1h", "GU5m", "SP5001h", "SP5005m"]:
        return SymbolicRegressionFitnessFunction.init_from_data_file(
            "../data/finance/" + key + "_gsgp.dat", split=0.7)

    elif key in ["GOLD1h_class", "GOLD5m_class", "GU1h_class", "GU5m_class", "SP5001h_class", "SP5005m_class"]:
        return SymbolicRegressionFitnessFunction.init_from_data_file(
            "../data/finance/" + key[:-6] + "_gsgp.dat", split=0.7,
            defn="classification_accuracy")


    # FIXME not sure how to implement this -- x[0] is a column,
    # so we can't take range(x[0])...
    # elif key == "keijzer-6":
    #     return SymbolicRegressionFitnessFunction.init_from_target_fn(
    #         lambda x: sum(1.0/i for i in range(x[0])),
    #         {"minv": [1], "maxv": [50], "incrv": [1]},
    #         test1={"minv": [1], "maxv": [120], "incrv": [1]})


    else:
        print "Unknown benchmark problem " + key


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: fitness.py <keyword>.")
        sys.exit()

    elif sys.argv[1] == "test_boolean":
        fn1 = "lambda x: ~(x[0] ^ x[1] ^ x[2] ^ x[3] ^ x[4])" # even-5 parity
        fn2 = "lambda x: True" # matches e5p for 16 cases out of 32
        fn3 = "lambda x: x[0] ^ x[1] ^ x[2] ^ x[3] ^ x[4]" # never matches e5p

        b = BooleanProblem(5, eval(fn1)) # target is e5p itself
        print(b(fn1)) # fitness = 0
        print(b(fn2)) # fitness = 16
        print(b(fn3)) # fitness = 32

        # Can also pass in a list of target values, ie semantics phenotype
        b = BooleanProblem(2, [False, False, False, False])
        print(b(fn2))

    elif sys.argv[1] == "test_random":
        fn1 = "dummy"
        r = RandomFitness()
        print(r(fn1))

    elif sys.argv[1] == "test_size":
        fn1 = "0123456789"
        fn2 = "01234567890123456789"
        s = SizeFitness(20)
        print(s(fn1))
        print(s(fn2))

    elif sys.argv[1] == "test_max":
        fn1 = "((0.5 + 0.5) + (0.5 + 0.5)) * ((0.5 + 0.5) + (0.5 + 0.5))"
        m = MaxFitness()
        print(m(fn1))

    elif sys.argv[1] == "test_sr":
        sr = benchmarks("vladislavleva-12")
        g = "lambda x: 2*x"
        print(sr(g))
        sr = benchmarks("identity")
        g = "lambda x: x"
        print(sr(g))

    elif sys.argv[1] == "test_pagie":
        sr = benchmarks("pagie-2d")
        g = "lambda x: (1 / (1 + x[0] ** -4) + 1 / (1 + x[1] ** -4))"
        print(sr(g))

    elif sys.argv[1] == "test_bioavailability":
        sr = benchmarks("vanneschi-bioavailability")
        g = "lambda x: x[0]*x[1]"
        print(sr(g))

    elif sys.argv[1] == "test_sr_mesh":
        test_build_mesh()
    elif sys.argv[1] == "test_sr_random":
        test_build_random()
