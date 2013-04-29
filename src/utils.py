#!/usr/bin/env python

import numpy as np
from fitness import SymbolicRegressionFitnessFunction as SRFF

def predict_constant(filename):
    d = np.genfromtxt(filename)
    # get the last column
    y = d.T[-1] 
    mn = np.mean(y)
    # generate a constant column
    pred = np.ones(len(y)) * mn 
    # calculate RMS error
    err = SRFF.rmse(y, pred)
    print("Predicting a constant %f gets error %f" % (mn, err))

def predict_linear_regression(filename):
    from sklearn.linear_model import LinearRegression
    d = np.genfromtxt(filename)
    # split into X (all columns up to last) and y (last column)
    X = d.T[:-1].T
    y = d.T[-1:].T
    lr = LinearRegression()
    lr.fit(X, y)
    pred = lr.predict(X)
    err = SRFF.rmse(pred.T[0], y.T[0])
    print("Linear Regression gets error %f" % (err,))
    
