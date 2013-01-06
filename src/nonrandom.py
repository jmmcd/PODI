#!/usr/bin/env python

import random, sys, itertools

class NonRandom(random.Random):
    """This is a class derived from and implementing the same
    interface as random.Random. The idea is, we construct it by
    passing in a deterministic iterator which generates integers in
    [0, sys.maxint]. We use that to generate numbers which, of course,
    are not random. An instance of this class can then be dropped in
    in place of the random module. For example, see "use_random"
    below, which runs fine whether random is the random module, or an
    instance of NonRandom. Instead of writing:

    import random
    x = random.randint(0, 10)

    we will now write:
    
    from nonrandom import NonRandom
    random = NonRandom(<iterator>)
    x = random.randint(0, 10)
    
    and everything else in the program can remain the same.

    Note that this is quite different from setting the seed of a
    rng. Although both situations will return a deterministic sequence
    of apparently random numbers, when using an rng and setting its
    seed, one can only vary *one* item independently -- all the
    remaining items are determined by it. In our situation, all the
    elements of the passed-in iterator can vary independently. That
    allows us to see the iterator as a genome on which we can run
    crossover and mutation. Note that the docs for random explain that
    to subclass Random(), one should override random(), seed(),
    getstate(), setstate() and jumpahead()
    [http://docs.python.org/library/random.html]. However, that is for
    true random number generators. If we override just random() to use
    the iterator as its source, all the other user-facing methods
    we're likely to want (randint, shuffle, choice, etc) will be
    affected by it. If the user does call seed(), getstate(),
    setstate(), or jumpahead(), something will happen internally in
    Random but it won't affect the user-facing calls. FIXME possibly
    should over-ride those methods to raise NotImplemented.

    If the passed-in iterator will only contain numbers in, say, [0,
    100], then the user should pass in maxval=100."""

    def __init__(self, it, maxval=sys.maxint, wraps=0):
        # Convert to iterator. No effect if already is an iterator.
        self.it = iter(it * (wraps + 1))
        self.maxval = float(maxval)
        self.used = 0
    def __new__(cls, it, maxval=sys.maxint, wraps=0):
        """We have to override __new__. For reasons. See
        [http://stackoverflow.com/questions/5148198/problem-subclassing-random-random-python-cpython-2-6]."""
        return super(NonRandom, cls).__new__(cls)
    def random(self):
        # An alternative implementation would generate floats in [0,
        # 1] and then avoid the divide. (In fact, same implementation
        # is fine: just pass maxval=1.0 and it will work.)
        self.used += 1
        return next(self.it) / self.maxval

if __name__ == "__main__":
    def use_random():
        print(random.random())
        print(random.random())
        print(random.random())
        print(random.randint(5, 12))
        print(random.randint(5, 12))
        print(random.randint(5, 12))
        x = [1, 2, 3, 4, 5]
        print(random.choice(x))
        random.shuffle(x)
        print(x)
        
    print("\nUsing random\n############")
    use_random()
    random = NonRandom(itertools.count(0), 20.0)
    print("\nUsing nonrandom\n###############")
    use_random()
