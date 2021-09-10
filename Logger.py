# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 10:25:10 2021

@author: aoust
"""


import pandas as pd
from datetime import date
import json

            
class LightStatLogger():
    
    def __init__(self,RSolver, instance_name):
        self.RSolver = RSolver
        self.init_log()
        self.date = str(date.today())
        self.name = instance_name
        self.configname = self.RSolver.config['name']
        with open("output/"+self.name+"_"+self.configname+"_"+self.date+"CONFIG_BUNDLE.txt", 'w+') as json_file:
            json.dump(RSolver.config, json_file)
        
    def saverule(self):
        return self.RSolver.it%100==99 or self.RSolver.finished 
    
    def init_log(self):
        self.it_log = []
        self.serious_step_log= []
        self.cut_added_log= []
        self.current_value_log= []
        self.best_value_log= []
        self.best_cert_value_log= []
        self.current_value_bar_log= []
        self.G_value_log = []
        self.delta_log= []
        self.gradnorm_log = []
        self.mu_log= []
        self.mbundle_log= []
        self.error_log= []
        self.qptime_log= []
        self.oracleTime_log= []
        self.bmTime_log = []
        self.concattime_log = []
        self.gramtime_log = []
        self.add_cut_time_log= []
        self.num_cuts_logs = []
        self.froebenius_log = []
        self.gershgorin_log = []
        self.Flinerrorshare_log = []
        
            
    def log(self):
        self.it_log.append(self.RSolver.it)
        self.serious_step_log.append(self.RSolver.serious_step_number)
        self.current_value_log.append(self.RSolver.current_value)
        self.best_value_log.append(self.RSolver.best_value)
        self.best_cert_value_log.append(self.RSolver.best_certified_value)
        self.current_value_bar_log.append(self.RSolver.current_value_bar)
        self.G_value_log.append(self.RSolver.Gval)
        self.delta_log.append(self.RSolver.delta)
        self.gradnorm_log.append(self.RSolver.grad_norm)
        self.mu_log.append(self.RSolver.kappa)
        self.mbundle_log.append(self.RSolver.mbundle)
        self.error_log.append(self.RSolver.error)
        self.gramtime_log.append(self.RSolver.gramtime)
        self.concattime_log.append(self.RSolver.concattime)
        self.qptime_log.append(self.RSolver.qptime)
        self.oracleTime_log.append(self.RSolver.oracleTime)
        self.bmTime_log.append(self.RSolver.bmtime)
        self.num_cuts_logs.append(len(self.RSolver.eigencuts_coefs))
        
        
        if self.saverule():
            self.DF = pd.DataFrame()
            self.DF["iterationsNumber"] =  self.it_log
            self.DF["seriousStepNumber"] =  self.serious_step_log
            self.DF["objectiveValue"] = self.current_value_log
            self.DF["objectiveValueBar"] = self.current_value_bar_log
            self.DF["bestObjectiveValue"] = self.best_value_log
            self.DF["bestcertifiedObjectiveValue"] = self.best_cert_value_log
            self.DF["mainObjective"] = self.G_value_log
            self.DF["delta"] = self.delta_log
            self.DF['gradnorm'] = self.gradnorm_log
            self.DF["mu"] = self.mu_log
            self.DF["mbundle"] = self.mbundle_log
            self.DF["minEigenvalue"] = self.error_log
            self.DF["qpTime"] = self.qptime_log
            self.DF["oracleTime"] = self.oracleTime_log
            self.DF["bmTime"] = self.bmTime_log
            self.DF["gramTime"] = self.gramtime_log
            self.DF["concaTime"] = self.concattime_log
            self.DF["num_cuts"] = self.num_cuts_logs
            self.DF.to_csv("output/"+self.name+"_"+self.configname+"_"+self.date+"LOG.csv")
      