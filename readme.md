# dualACOPFsolver
Implementation of a proximal bundle method (PBM) to solve the ACOPF's dual problem, which is formulated as an unconstrained concave maximisation
problem with a partially separable structure induced by the clique decomposition.
In particular, we use this PBM as a processing step to polish MOSEK's dual solution.

# Dependencies

Required python packages:
- numpy
- scipy
- pandas
- osqp
- chompack 

# Test instances

The ACOPF instances are taken from the library PGLib (https://github.com/power-grid-lib/pglib-opf), which is maintained by the IEEE PES Task Force on Benchmarks for Validation of Emerging Power System Algorithms.

# Numerical experiments

Executing the file main.py will run the numerical experiments presented in the paper "A. Oustry, C. D'Ambrosio, L. Liberti, M. Ruiz, The dual ACOPF as a sparse and unconstrained nonsmooth optimization problem" (under review). Executing the file stats.py produces the full result table.

---------------------------------------------------------------------------------------
Researchers affiliated with

(o) LIX CNRS, École polytechnique, Institut Polytechnique de Paris, 91128, Palaiseau, France 

(o) École des Ponts, 77455 Marne-La-Vallée

---------------------------------------------------------------------------------------

Sponsored by Réseau de transport d’électricité, 92073 La Défense, France




