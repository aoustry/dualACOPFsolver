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

The ACOPF instances are taken from the PGLib (https://github.com/power-grid-lib/pglib-opf).

---------------------------------------------------------------------------------------
Researchers affiliated with

(o) LIX CNRS, École polytechnique, Institut Polytechnique de Paris, 91128, Palaiseau, France 

(o) École des Ponts, 77455 Marne-La-Vallée

---------------------------------------------------------------------------------------

Sponsored by Réseau de transport d’électricité, 92073 La Défense, France




