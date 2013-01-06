#!/usr/bin/env python

import random

idx = 0
nstates = random.randint(1, 5)
states = [{} for i in range(nstates)]
for state in states:
    degree = random.randint(2, 4)
    for i in range(degree):
        label = random.randint(0, 1)
        dest = random.randint(0, nstates-1)
        state[label] = dest

input = [0, 1, 0, 1, 0, 1]
for x in input:
    state = states[idx]
    try:
        idx = state[x]
    except KeyError:
        print("Failed to accept")
        break
    
    
