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
import dualACOPFsolver

lineconstraints = True
threshold_aggregation = 600
m = 0.01
rel_tol = 1E-6
ratio_added_cuts = 0.7
verbose = True
onepercent = 0.01
oneperthousand = 0.001
scaling_lambda = 1


def execute_bundle_solver_with_ws(name_instance):
    """Function to execute the proximal bundle algorithm as post-processing for MOSEK """
    np.random.seed(10)
    maxit = 500
    instance_config = {"lineconstraints" : lineconstraints,  "cliques_strategy":"ASP"}
    Instance = instance.ACOPFinstance("pglib-opf/{0}.m".format(name_instance),name_instance,instance_config)
    
    ACOPFsolver_config  = {'name':'config_with_ws',
              "rel_tol":rel_tol,
              'mbundle' : m,
              'maxit' : maxit, 
              "scaling_lambda" : scaling_lambda, 
              'aggregation':Instance.n>threshold_aggregation, 
              'ratio_added_cuts' : ratio_added_cuts,
              }
    B = mosekACOPFsolver.MosekRelaxationSolver(Instance)
    t0 = time.time()
    print('Mosek starts.')
    mosek_claimed_value,alpha, beta, gamma, lamda_f, lamda_t, nu = B.solve()
    mosek_time =  time.time() - t0
    print('Mosek terminated.')
    R = dualACOPFsolver.dualACOPFsolver(Instance, ACOPFsolver_config)
    R.loggers = [LightStatLogger(R,name_instance)]
    R.verbose = verbose
    kappa = R.n * oneperthousand
    mosek_value = R.certified_value(alpha, beta, gamma, ACOPFsolver_config['scaling_lambda']*lamda_f, ACOPFsolver_config['scaling_lambda']*lamda_t,nu)
    
    with open("output/"+name_instance+"_"+"_"+"MOSEK_VALS.txt", 'w+') as txt_file:
        txt_file.write(str(mosek_claimed_value)+","+str(mosek_value)+","+str(mosek_time))
        txt_file.close()
    
    R.set_inital_values(alpha, beta, gamma, ACOPFsolver_config['scaling_lambda']*lamda_f, ACOPFsolver_config['scaling_lambda']*lamda_t,nu)
    R.solve(kappa)
    return R

def execute_bundle_solver(name_instance):
    """Function to execute the proximal bundle algorithm with no warmstart (standalone solver)"""
    np.random.seed(10)
    maxit = 3000
    instance_config = {"lineconstraints" : lineconstraints,  "cliques_strategy":"ASP"}
    Instance = instance.ACOPFinstance("pglib-opf/{0}.m".format(name_instance),name_instance,instance_config)
    ACOPFsolver_config  = {'name':'config_no_ws',
              "rel_tol":rel_tol,
              'mbundle' : m,
              'maxit' : maxit, 
              "scaling_lambda" : scaling_lambda, 
              'aggregation':Instance.n>threshold_aggregation, 
              'ratio_added_cuts' : ratio_added_cuts,
              }
    R = dualACOPFsolver.dualACOPFsolver(Instance, ACOPFsolver_config)
    R.loggers = [LightStatLogger(R,name_instance)]
    R.verbose = verbose
    kappa = R.n * oneperthousand
    R.solve(kappa)
    return R


for file in instances:
    name_instance = file[:-2]
    print(name_instance)
    try:
        execute_bundle_solver_with_ws(name_instance)
    except Exception as e:
          with open("output/"+name_instance+"_"+"_"+"ERROR.txt", 'w+') as txt_file:
              txt_file.write(str(e))
              txt_file.close()


