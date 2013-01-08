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

Abstract: For many real-world problems, there exist non-deterministic
heuristics which generate valid but possibly sub-optimal
solutions. The *program optimisation with dependency injection*
method, introduced here, allows such a heuristic to be placed under
evolutionary control, allowing search for the optimum. Essentially,
the heuristic is "fooled" into using a genome, supplied by a genetic
algorithm, in place of the output of its random number generator. The
method is demonstrated with generative heuristics in the domains of 3D
design and communications network design. It is also used in novel
approaches to genetic programming.

Authors: James McDermott and Paula Carroll, Management Information
Systems, Quinn School of Business, University College Dublin, Ireland.
