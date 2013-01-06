#!/usr/bin/env python

import variga
import numpy as np

def generate_list(random):
    """A stupid fitness function: tries to pick 4 integers from [0-9]
    such that their product is large."""
    s = list(range(10))
    return random.sample(s, 4)

variga.GENERATE = generate_list
variga.FITNESS = lambda x: np.product(x)
variga.SUCCESS = lambda x: x > 2500
variga.MAXIMISE = True
variga.main()
