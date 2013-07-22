#!/usr/bin/env python

import variga
import numpy as np

class fitness:
    def __call__(self, x):
        return self.get_semantics(x)[0]
    def test(self, x):
        return self(x)
    def get_semantics(self, x):
        p = np.product(x)
        return p, x

def generate_list(random):
    """A stupid problem: tries to pick 4 integers from [0-9]
    such that their product is large."""
    s = list(range(10))
    x = random.sample(s, 4)
    # return readable_phenotype, phenotype
    return x, x

variga.GENERATE = generate_list
variga.FITNESS = fitness()
variga.SUCCESS = lambda x: x > 2500
variga.MAXIMISE = True
variga.main()
