# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 20:29:43 2021

@author: aoust
"""

import pandas as pd
import numpy as np
import os
from scipy.stats import gmean

'''Building results tables for MOSEK's post-processing by the PBM '''

folder = "output/"

def extract_number(s):
    char = [str(i) for i in range(10)]
    res = ''
    for el in s:
        if el in char:
            res = res+el
    return int(res)

def compute_stats_instance_bundle(path,maxlength):
    dataframe = pd.read_csv(path)
    N = min(len(dataframe),maxlength)
    best_obj = max(dataframe["bestcertifiedObjectiveValue"][:N])
    average_iteration_time = dataframe["qpTime"][:N].mean()+ dataframe["oracleTime"][:N].mean() + dataframe["bmTime"][:N].mean() + dataframe["gramTime"][:N].mean()+ dataframe["concaTime"][:N].mean()
    percentage_QP = 100*dataframe["qpTime"][:N].mean()/average_iteration_time
    return N, best_obj , average_iteration_time ,percentage_QP
    
def compute_stats_instance_mosek(path):
    metrics = list((pd.read_csv(path)).columns)
    return [float(s) for s in metrics]

def compute_mosek_status(path):
    f = open(path)
    for line in f.readlines():
        for el in  ['UNKNOWN','OPTIMAL'] :
            if el in line:
                return el.lower()
    return np.nan

def compute_stats_instance_mips(path):
    if int(list((pd.read_csv(path+"_success.csv")).columns)[0]):
        
        return float(list((pd.read_csv(path+"_obj.csv")).columns)[0])
    return np.nan

def compute_table(maxlength):
    names, Ns, best_objs, it_times, percentages = [],[],[],[],[]
    mosek_evs,mosek_cvs,mosek_times = [],[],[]
    ubs,status = [],[]
    size = []
    for file in os.listdir(folder):
        if ".csv" in file and not("mosek" in file):
            instance_name = file.split('_config_')[0]
            size.append(extract_number(instance_name))
            N,best_obj, it_time, percentage_oracle = compute_stats_instance_bundle(folder+file,maxlength)
            mosek_ev,mosek_cv,mosek_time = compute_stats_instance_mosek(folder+instance_name+"__MOSEK_VALS.txt")
            names.append(instance_name)
            Ns.append(N)
            best_objs.append(best_obj)
            it_times.append(it_time)
            percentages.append(percentage_oracle)
            mosek_cvs.append(mosek_cv)
            mosek_evs.append(mosek_ev)
            mosek_times.append(mosek_time)
            ubs.append(compute_stats_instance_mips('ipopt_output/'+instance_name))
            status.append(compute_mosek_status(folder+instance_name+"_mosek.txt"))
    df = pd.DataFrame()
    ######MOSEK INFO
    df["Instance"] = names
    df['n'] = size
    df['Estimated LB (Mosek)'],df['Certified LB (Mosek)'],df['CPU time (Mosek)'] =  mosek_evs,mosek_cvs,mosek_times
    df['mosek_status'] = status
    ######Bundle INFO
    df["PBM It. nbr"], df['PBM avg. it. time'] = Ns, it_times
    df['overhead'] =  (df["PBM It. nbr"]* df['PBM avg. it. time'])/df['CPU time (Mosek)']
    df["Certified LB (PBM)"] = best_objs
    
    df['Progress wrt MOSEK estimated LB (%)'] = 100*(df["Certified LB (PBM)"]-df['Estimated LB (Mosek)'])/df['Estimated LB (Mosek)']
    
    df['Progress wrt MOSEK certified LB (%)'] = 100*(df["Certified LB (PBM)"]-df['Certified LB (Mosek)'])/df['Certified LB (Mosek)']
    
    ######MIPS and gap
    df['IPOPT UB'] = ubs
    df['Reduction of certified gap (%)'] = np.maximum(np.zeros(len(df)),100*(df["Certified LB (PBM)"]-df['Certified LB (Mosek)'])/(df['IPOPT UB']-df['Certified LB (Mosek)']))
    
    df['Reduction of estimated gap (%)'] = np.maximum(np.zeros(len(df)),100*(df["Certified LB (PBM)"]-df['Estimated LB (Mosek)'])/(df['IPOPT UB']-df['Estimated LB (Mosek)']))

    df = df.sort_values(by=['n'])
    return df 



def store_clean_table_accuracy_progress(df):
    table = pd.DataFrame()
    table['Instance'] = [s.replace('pglib_opf_case','') for s in df['Instance']]
    table['Mosek EV'] = [("%.6g" % best_obj) for best_obj in df['Estimated LB (Mosek)']]
    table['Mosek CV'] = [("%.6g" % best_obj) for best_obj in df['Certified LB (Mosek)']]
    table['Bundle CV'] = [("%.6g" % best_obj) for best_obj in df['Certified LB (PBM)']]
    table['Progress wrt EV'] = [("%.2g" % best_obj)+"%" for best_obj in df['Progress wrt MOSEK estimated LB (%)']]
    table['Progress wrt CV'] = [("%.2g" % best_obj)+"%" for best_obj in df['Progress wrt MOSEK certified LB (%)']]
    table['aux'] = [x for x in df['Progress wrt MOSEK certified LB (%)']]
    table = table.sort_values(by=['aux'],ascending=False)
    print((table[['Instance','Mosek EV','Mosek CV','Bundle CV','Progress wrt EV','Progress wrt CV']]).to_latex("stats_acc_progress.txt", index = False))


    
def store_clean_table_gap_progress(df):
    table = pd.DataFrame()
    table['Instance'] = [s.replace('pglib_opf_case','') for s in df['Instance']]
    table['UB'] = [("%.6g" % best_obj) for best_obj in df['IPOPT UB']]
    table['Cert. gap reduction'] = [("%.2g" % best_obj)+"%" for best_obj in df['Reduction of certified gap (%)']]
    table['Est. gap reduction'] = [("%.2g" % best_obj)+"%" for best_obj in df['Reduction of estimated gap (%)']]
    table['Time overhead'] = [("%.4g" % x) for x in df['overhead']]
    table = table.dropna()
    table['aux'] = [x for x in df['Reduction of certified gap (%)']]
    table = table.sort_values(by=['aux'],ascending=False)
    print(table[['Instance', 'UB','Est. gap reduction','Cert. gap reduction','Time overhead']].to_latex("stats_gap_progress.txt", index = False))

def display_agg_stats(df):
    print('Average overhead',gmean(df['overhead']))
        
    print("""Progress wrt estimated LB """)
    print("0.5%",np.sum(df['Progress wrt MOSEK estimated LB (%)']>0.5))
    print("0.2%",np.sum(df['Progress wrt MOSEK estimated LB (%)']>0.2))
    print("0.1%",np.sum(df['Progress wrt MOSEK estimated LB (%)']>0.1))
    
    print("""Progress of certified LB """)
    print("0.5%",np.sum(df['Progress wrt MOSEK certified LB (%)']>0.5))
    print("0.2%",np.sum(df['Progress wrt MOSEK certified LB (%)']>0.2))
    print("0.1%",np.sum(df['Progress wrt MOSEK certified LB (%)']>0.1))
    
    print("""Reduction estimated gap -> certified gap """)
    print("80%",np.sum(df['Reduction of estimated gap (%)']>80))
    print("50%",np.sum(df['Reduction of estimated gap (%)']>50))
    print("20%",np.sum(df['Reduction of estimated gap (%)']>20))
    
    print("""Reduction of certified gap """)
    print("80%",np.sum(df['Reduction of certified gap (%)']>80))
    print("50%",np.sum(df['Reduction of certified gap (%)']>50))
    print("20%",np.sum(df['Reduction of certified gap (%)']>20))
    

df = compute_table(500)

with open('Table of results (generated by stats.py).txt','w') as f:
    f.write(df.to_markdown())
f.close()

display_agg_stats(df[df['n']>=1000])
table = store_clean_table_accuracy_progress(df[df['n']>=1000])
store_clean_table_gap_progress(df[df['n']>=1000])
# df.to_csv("stats.csv", index = False)
# print(df.to_latex("stats.txt",index = False))


