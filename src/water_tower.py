#!/usr/bin/env python

"""
Water Tower

Parametric model of a UCD-style Water Tower

TODO: allow if-statements in the numerical functions.
TODO: check the possible values and interactions of height and radius functions.

"""

import graph
from math import *
import sys, os

def grow(rng, max_depth):
    if max_depth == 0:
        return rng.choice(["t", "(1.0 / pi)", "1.0", "pi"])
    else:
        return rng.choice(["sin("+grow(rng, max_depth-1)+")",
                           "("+grow(rng, max_depth-1)+"+"+grow(rng, max_depth-1)+")",
                           "("+grow(rng, max_depth-1)+"-"+grow(rng, max_depth-1)+")",
                           "("+grow(rng, max_depth-1)+"*"+grow(rng, max_depth-1)+")"])

def make_arithmetic_fn(rng, max_depth):
    s = grow(rng, max_depth)
    f = lambda t: eval(s)
    return f

class WaterTower(graph.graph):

    def __init__(self):
        super(WaterTower, self).__init__()

    def run(self, rng):
        self.rng = rng
        self.rad = rng.random() * 3.0
        self.freq = rng.random() * 2.0 * pi
        self.const = rng.random() * 1.5
        self.vary_height_amount = rng.random() * 0.5
        # do our stuff here
        # Try altering any of these parameters for new towers!
        self.nnodes = rng.randint(3, 9)
        self.nlayers = rng.randint(2, 9)
        possible_offsets = [1, self.nnodes]
        if self.nnodes % 2 == 0:
            possible_offsets.append(self.nnodes / 2)
        if self.nnodes % 3 == 0:
            possible_offsets.append(self.nnodes / 3)
        noffsets = self.rng.choice(possible_offsets)
        self.offsets = [self.rng.randint(0, 1) * 0.5 for i in range(noffsets)]

        possible_connections = [(0,), (-1, 0), (-1, 1), (0, 1)]
        connections = self.rng.choice(possible_connections)
        self.i2self = (0 in connections) # connect to corresponding node in previous layer
        self.i2selfm1 = (-1 in connections) # connect to previous node in previous layer
        self.i2selfp1 = (1 in connections) # connect to next node in previous layer

        self.layers = []
        self.height = self.rng.random() * 3.0
        self.uniform = bool(rng.randint(0, 1))
        self.vary_height_within_layer = bool(rng.randint(0, 1))
        # should every second layer be doubled nnodes? 1 for no, 2 for yes
        self.doublep = self.rng.randint(1, 2)
        self.join_entire_layer = True
        self.phi = self.rng.random() * pi
        self.radius = make_arithmetic_fn(self.rng, 2)

        for j in range(self.nlayers):
            self.add_layer(j, self.doublep, self.join_entire_layer)

        for i in range(self.nlayers):
            self.phi += self.offsets[i % len(self.offsets)]
            for node in self.layers[i]:
                idx = int(self.node[node]["idx"])
                r = self.radius(i / float(len(self.layers)))

                theta = (self.phi + idx) / float(len(self.layers[i]))
                # print("idx, phi, theta = " + str(idx) + " " + str(self.phi) + " " + str(theta))
                z = self.height * i + self.height_fn(i)
                if len(self.layers[i]) % 2 == 0 and self.vary_height_within_layer:
                    if idx % 2 == 1:
                        z += self.vary_height_amount * (idx % 2)
                xyz = (
                    r * cos(2 * pi * theta),
                    r * sin(2 * pi * theta),
                    z)
                self.node[node]["xyz"] = xyz
                self.node[node]["x"] = xyz[0]
                self.node[node]["y"] = xyz[1]
                self.node[node]["z"] = xyz[2]

        return self

    def add_layer(self, layer_idx, doublep=1, join_entire_layer=True):

        # print("layer_idx = " + str(layer_idx))
        n = self.order()

        if doublep == 2:
            if layer_idx % 2 == 0:
                len_layer = self.nnodes
                len_prev = 2 * self.nnodes
            else:
                len_layer = 2 * self.nnodes
                len_prev = self.nnodes
        else:
            len_layer = self.nnodes

        this_layer = xrange(n, n + len_layer)
        if layer_idx == 0:
            prev_layer = [] # will be unused
        else:
            prev_layer = self.layers[layer_idx - 1]
        self.layers.append(this_layer)

        for i, node in enumerate(this_layer):
            # add nodes of this layer
            # print("adding node " + str(node))
            self.add_node(node, layer=str(layer_idx), idx=str(i))

            # add edges within layer
            if self.join_entire_layer or (i % 2 == 0):
                next_idx = (i + 1) % len(this_layer)
                self.add_edge(this_layer[i], this_layer[next_idx])

            # add edges to previous layer
            if layer_idx == 0:
                continue
            if doublep == 1:
                if self.i2self:
                    self.add_edge(this_layer[i], prev_layer[i])
                if self.i2selfm1:
                    self.add_edge(this_layer[i], prev_layer[(i-1)%self.nnodes])
                if self.i2selfp1:
                    self.add_edge(this_layer[i], prev_layer[(i+1)%self.nnodes])
            else:
                if layer_idx % 2 == 0:
                    if self.i2self:
                        self.add_edge(this_layer[i], prev_layer[(2*i)%len_prev])
                    if self.i2selfm1:
                        self.add_edge(this_layer[i], prev_layer[(len_prev+2*i-1)%len_prev])
                    if self.i2selfp1:
                        self.add_edge(this_layer[i], prev_layer[(2*i+1)%len_prev])
                if layer_idx % 2 == 1:
                    if self.i2self:
                        src, dest = this_layer[i], prev_layer[i/2]
                        self.add_edge(src, dest)
                    if self.i2selfm1:
                        src, dest = this_layer[i], prev_layer[(len_prev+i/2-1)%len_prev]
                        self.add_edge(src, dest)
                    if self.i2selfp1:
                        src, dest = this_layer[i], prev_layer[(i/2+1)%len_prev]
                        self.add_edge(src, dest)

    # # alter this function to change how height depends on layer index
    def height_fn(self, i):
        return self.const + self.rad * sin(self.freq * i)



# This global constant should be made available by any generative
# model so that architype can know what class to run
MODEL = WaterTower

if __name__ == "__main__":
    g = WaterTower()
    print(g.node)
