# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 16:41:55 2021

@author: aoust
"""

import instance, time
import numpy as np
from Logger import LightStatLogger
from cases_list import instances
import mosekACOPFsolver
import dualACOPFsolverSmoothing

name_instance = 'pglib_opf_case118_ieee'
lineconstraints = False
instance_config = {"lineconstraints" : lineconstraints,  "cliques_strategy":"ASP"}
Instance = instance.ACOPFinstance("pglib-opf-TYP/{0}.m".format(name_instance),name_instance,instance_config)   

B = mosekACOPFsolver.MosekRelaxationSolver(Instance)
mosek_claimed_value,alpha, beta, gamma, lamda_f, lamda_t, nu = B.solve()
print("Mosek claimed value = {0}".format(mosek_claimed_value))

R = dualACOPFsolverSmoothing.dualACOPFsolver(Instance)
#R.test()
R.solve(100,1e-4)
print('Règle pour updater thetaref. Plus fréquemment?')
print("Updater mu? Le réduire progressivement")
print('Lien entre mu, epsilon et aussi le critère darret interne. Par rapport à la pénal smoothing et/ou par rapport à lalgo newton')
print('critere darret final')