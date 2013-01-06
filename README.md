Program Optimisation with Dependency Injection (PODI)
=====================================================

PODI is a highly general evolutionary algorithm. It can be used as a
normal genetic algorithm, or to carry out genetic programming or
grammatical evolution, or to attack problems like TSP which require
alternative representations. It achieves this generality using the
same idea as GE: a variable-length integer array genome is evolved by
a GA, and the genotype to phenotype mapping process consists of
reading the genome one integer at a time, each one making one
"decision" in the creation of the phenotype. The difference is that
instead of the fixed grammar derivation process of GE, in PODI *any*
non-deterministic program can be used. That program's possible outputs
are the feasible solution space, ie the possible phenotypes. We can
then see that non-deterministic program (NDP) as a mapping from
integer-array genotype to phenotype.

This repository contains Python source code implementing the PODI
idea, and several example NDPs. These examples allow PODI to emulate
GE precisely, and to carry out two novel forms of GP (no claim is made
as yet that they are efficient). There is also code for running most
of the experiments in the EuroGP2013 paper (see below). Code for
running the RSAP experiment in that paper is not provided since the
heuristic NDP used there is not under the copyright of the author.

If you wish to cite this project, please cite this paper:

McDermott and Carroll, *Program Optimisation with Dependency
Injection*, in *Proceedings of EuroGP 2013*, Vienna, Austria,
Springer.

Abstract: A highly general evolutionary algorithm is introduced, named
*program optimisation with dependency injection*. It can be used to
search among the possible outputs of any non-deterministic program, as
follows. The program's random number generator is replaced by a
non-random number generator: this is dependency injection. The
non-random numbers are supplied by genomes under the control of a
genetic algorithm. The program can then be seen as a genome-phenotype
mapping. It is demonstrated that suitable programs include the
derivation program used in grammatical evolution, two tree-growing
functions useful for genetic programming, and generative programs in
the domains of 3D design and communications network design.

Authors: James McDermott and Paula Carroll, Management Information
Systems, Quinn School of Business, University College Dublin, Ireland.
