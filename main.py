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

name_instance = 'pglib_opf_case89_pegase'
lineconstraints = False
instance_config = {"lineconstraints" : lineconstraints,  "cliques_strategy":"ASP"}
Instance = instance.ACOPFinstance("pglib-opf-TYP/{0}.m".format(name_instance),name_instance,instance_config)   
R = dualACOPFsolverSmoothing.dualACOPFsolver(Instance)
R.test()
