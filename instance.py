# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 09:53:00 2021

@author: aoust
"""

import time, itertools, operator,osqp, parserOPF
import numpy as np
from scipy.sparse import lil_matrix,  coo_matrix, identity,diags, csc_matrix, hstack,vstack,diags
from scipy.sparse import linalg as sla
from scipy.sparse.linalg import eigsh as sparse_eighs
import scipy.sparse.linalg as linalg
from cvxopt import spmatrix, amd
import chompack as cp
import random
import networkx as nx


#My infty
myinf_power_lim = 1E4

class ACOPFinstance():
    
    def __init__(self, filepath,name,config):
        """
        
        Parameters
        ----------
        filepath : string
            Location of the .m file in MATPOWER data format.

        -------
        Load the model.

        """
       
        print("Start loading instance " +name)
        self.name = name
        self.config = config
        
        parser = parserOPF.mpcCase(filepath) 
        OPF_instance = parserOPF.OPF_Data(parser)
        
        #Sizes
        self.baseMVA = OPF_instance.baseMVA
        self.n, self.m, self.gn = OPF_instance.n, OPF_instance.m, OPF_instance.gn
        #Generator quantities
        self.C = OPF_instance.C
        genlist,lincost, quadcost = [],[],[]
        for generator in OPF_instance.SLR:
            genlist.append(generator)
            lincost.append(self.C[(generator[0],generator[1],1)])
            quadcost.append(self.C[(generator[0],generator[1],2)])
        self.lincost = np.array(lincost)
        self.quadcost = np.array(quadcost)
        self.genlist = genlist
        assert(len(self.genlist)==self.gn)
        self.Pmin, self.Qmin, self.Pmax, self.Qmax = [OPF_instance.SLR[self.genlist[idx_gen]] for idx_gen in range(self.gn)], [OPF_instance.SLC[self.genlist[idx_gen]] for idx_gen in range(self.gn)], [OPF_instance.SUR[self.genlist[idx_gen]] for idx_gen in range(self.gn)], [OPF_instance.SUC[self.genlist[idx_gen]] for idx_gen in range(self.gn)]
        # self.index_quad_generators = set()
        # for i in range(len(self.quadcost)):
        #     if abs(self.quadcost[i]) >= myZeroforCosts:
        #         self.index_quad_generators.add(i)
        self.offset = 0
        for generator in OPF_instance.SLR:
            if (generator[0],generator[1],0) in self.C:
                self.offset+=self.C[(generator[0],generator[1],0)]
        
        #Bus quantities
        self.buslist = []
        self.buslistinv ={}
        i=0
        for bus in OPF_instance.busType:
            self.buslist.append(bus)
            self.buslistinv[bus] = i
            i+=1
        for i in range(self.n):
            assert(self.buslistinv[self.buslist[i]]==i)
        self.busType = OPF_instance.busType
        self.Vmin, self.Vmax = [OPF_instance.VL[self.buslist[i]] for i in range(self.n)], [OPF_instance.VU[self.buslist[i]] for i in range(self.n)]
        self.A = OPF_instance.A
        self.Pload = [np.real(OPF_instance.SD[self.buslist[i]]) for i in range(self.n)]
        self.Qload = [np.imag(OPF_instance.SD[self.buslist[i]]) for i in range(self.n)]
        self.preprocessing_power_bounds()
                
        
        #Lines quantities
        self.status = OPF_instance.status
        self.Yff, self.Yft, self.Ytf, self.Ytt = OPF_instance.Yff, OPF_instance.Yft, OPF_instance.Ytf, OPF_instance.Ytt 
        self.cl = 0
        
        
        self.clinelist,self.clinelistinv = [],[]
        
        if self.config["lineconstraints"]:
            self.clinelist = []
            self.clinelistinv ={}
            i=0
            counter = 0
            for line in OPF_instance.SU:
                if OPF_instance.SU[line]>0:
                    self.clinelist.append(line)
                    self.clinelistinv[line] = i
                    i+=1 
                counter+=1
            self.cl = len(self.clinelist)
            
            
        self.Imax = [OPF_instance.SU[self.clinelist[idx_line]] for idx_line in range(self.cl)]
        self.test_status()
        
        #Construct cliques
        self.build_cliques(config["cliques_strategy"])
        self.SVM = {idx_clique : sum([self.Vmax[bus]**2 for bus in self.cliques[idx_clique]]) for idx_clique in range(self.cliques_nbr)}
        
                
        # #Construct m_cb matrices
        self.M = {}
        #Parts of M related to the lines #self.M[bus] = lil_matrix((self.n,self.n),dtype = np.complex128)
        for (b,a,h) in self.Yff:
            index_bus_b,index_bus_a = self.buslistinv[b],self.buslistinv[a]
            i,j = max(index_bus_b,index_bus_a),min(index_bus_b,index_bus_a)
            cliqueff = self.edges_to_clique[(i,j)]
            if not((cliqueff,index_bus_b) in self.M):
                nc = self.ncliques[cliqueff]
                self.M[cliqueff,index_bus_b] = lil_matrix((nc,nc),dtype = np.complex128)
            local_index_bus_b, local_index_bus_a = self.localBusIdx[cliqueff,index_bus_b],self.localBusIdx[cliqueff,index_bus_a]
            assert(local_index_bus_b!=local_index_bus_a)
            self.M[cliqueff,index_bus_b][local_index_bus_b,local_index_bus_b] += self.Yff[(b,a,h)]
            
                
        del cliqueff, local_index_bus_b, local_index_bus_a
        for (b,a,h) in self.Yft:
            index_bus_b,index_bus_a = self.buslistinv[b],self.buslistinv[a]
            i,j = max(index_bus_b,index_bus_a),min(index_bus_b,index_bus_a)
            cliqueft = self.edges_to_clique[(i,j)]
            if not((cliqueft,index_bus_b) in self.M):
                nc = self.ncliques[cliqueft]
                self.M[cliqueft, index_bus_b] = lil_matrix((nc,nc),dtype = np.complex128)
            local_index_bus_b, local_index_bus_a = self.localBusIdx[cliqueft,index_bus_b],self.localBusIdx[cliqueft,index_bus_a]
            assert(local_index_bus_b!=local_index_bus_a)
            self.M[cliqueft,index_bus_b][local_index_bus_b,local_index_bus_a] += self.Yft[(b,a,h)]
            
                
        del cliqueft, local_index_bus_b, local_index_bus_a
        for (a,b,h) in self.Ytt:
            index_bus_b,index_bus_a = self.buslistinv[b],self.buslistinv[a]
            i,j = max(index_bus_b,index_bus_a),min(index_bus_b,index_bus_a)
            cliquett = self.edges_to_clique[(i,j)]
            if not((cliquett, index_bus_b) in self.M):
                nc= self.ncliques[cliquett]
                self.M[cliquett,index_bus_b] =lil_matrix((nc,nc),dtype = np.complex128)
            local_index_bus_b, local_index_bus_a = self.localBusIdx[cliquett,index_bus_b],self.localBusIdx[cliquett,index_bus_a]
            assert(local_index_bus_b!=local_index_bus_a)
            self.M[cliquett,index_bus_b][local_index_bus_b, local_index_bus_b] += self.Ytt[(a,b,h)]
            
        del cliquett, local_index_bus_b, local_index_bus_a
        for (a,b,h) in self.Ytf:
            index_bus_b,index_bus_a = self.buslistinv[b],self.buslistinv[a]
            i,j = max(index_bus_b,index_bus_a),min(index_bus_b,index_bus_a)
            cliquetf = self.edges_to_clique[(i,j)]
            if not((cliquetf,index_bus_b) in self.M):
                nc= self.ncliques[cliquetf]
                self.M[cliquetf,index_bus_b] = lil_matrix((nc,nc),dtype = np.complex128)
            local_index_bus_b, local_index_bus_a = self.localBusIdx[cliquetf,index_bus_b],self.localBusIdx[cliquetf,index_bus_a]
            assert(local_index_bus_b!=local_index_bus_a)
            self.M[cliquetf,index_bus_b][local_index_bus_b,local_index_bus_a] += self.Ytf[(a,b,h)]
            
        del cliquetf, local_index_bus_b, local_index_bus_a, index_bus_b
        #Parts of M related to the shunts
        aux,test_sum = {},0
        for clique,index_bus in self.M:
            if not(index_bus in aux):
                test_sum+=1
                aux[index_bus] = 1
                local_index_bus = self.localBusIdx[clique,index_bus]
                self.M[clique,index_bus][local_index_bus,local_index_bus] += self.A[self.buslist[index_bus]]
        assert(test_sum==self.n)
        del aux, test_sum, clique,local_index_bus
        
               
        #Conversion to csc_matrices
        for couple in self.M:
            self.M[couple] = self.M[couple].tocsc()
            
        
        self.HM, self.ZM, self.assigned_buses, self.assigned_lines = {} , {},{},{}
        for idx_clique in range(self.cliques_nbr):
            self.assigned_buses[idx_clique] = set()
            self.assigned_lines[idx_clique] = set()
        del idx_clique
            
        for couple in self.M:
            self.HM[couple] = 0.5 * (self.M[couple]+(self.M[couple]).H)
            self.ZM[couple] = 0.5 * (self.M[couple]-(self.M[couple]).H)
            clique,idx_bus = couple
            self.assigned_buses[clique].add(idx_bus)
        del clique,idx_bus
        self.Nf = {}
        self.Nt = {}
        
        #Build Nf and Nt matrices
        if self.config["lineconstraints"]:
            
            for idx_line,line in enumerate(self.clinelistinv):
                b,a,h = line
                index_bus_b,index_bus_a = self.buslistinv[b],self.buslistinv[a]
                i,j = max(index_bus_b,index_bus_a),min(index_bus_b,index_bus_a)
                clique = self.edges_to_clique[(i,j)]
                nc = self.ncliques[clique]
                local_index_bus_b,local_index_bus_a = self.localBusIdx[clique,index_bus_b],self.localBusIdx[clique,index_bus_a]
                assert(local_index_bus_b!=local_index_bus_a)
                self.assigned_lines[clique].add(idx_line)
                #Build Nf line matrix
                self.Nf[clique,idx_line] = lil_matrix((nc,nc),dtype = np.complex128)
                self.Nf[clique,idx_line][local_index_bus_b,local_index_bus_b] = np.conj(self.Yff[line]) * self.Yff[line]
                self.Nf[clique,idx_line][local_index_bus_a,local_index_bus_b] = np.conj(self.Yft[line]) * self.Yff[line]
                self.Nf[clique,idx_line][local_index_bus_b,local_index_bus_a] = np.conj(self.Yff[line]) * self.Yft[line]
                self.Nf[clique,idx_line][local_index_bus_a,local_index_bus_a] = np.conj(self.Yft[line]) * self.Yft[line]
                
                #Build Nt line matrix
                self.Nt[clique,idx_line] = lil_matrix((nc,nc),dtype = np.complex128)
                self.Nt[clique,idx_line][local_index_bus_b,local_index_bus_b] = np.conj(self.Ytf[line]) * self.Ytf[line]
                self.Nt[clique,idx_line][local_index_bus_a,local_index_bus_b] = np.conj(self.Ytt[line]) * self.Ytf[line]
                self.Nt[clique,idx_line][local_index_bus_b,local_index_bus_a] = np.conj(self.Ytf[line]) * self.Ytt[line]
                self.Nt[clique,idx_line][local_index_bus_a,local_index_bus_a] = np.conj(self.Ytt[line]) * self.Ytt[line] 
                
                
    def build_cliques(self,strategy):
        self.edges = {}
        I = [i for i in range(self.n)]
        J = [i for i in range(self.n)]
        
        if strategy == "ASP":
            for (b,a,h) in self.Yff:
                index_bus_b,index_bus_a = self.buslistinv[b],self.buslistinv[a]
                i, j = max(index_bus_b,index_bus_a),min(index_bus_b,index_bus_a)
                if not((i, j) in self.edges):
                    self.edges[(i, j)]=1
                    I.append(i)
                    J.append(j)
                
        A = spmatrix(1.0, I, J, (self.n,self.n))
        symb = cp.symbolic(A, p=amd.order)
        self.cliques = (symb.cliques(reordered=False))
        for cl in self.cliques:
            cl.sort()
        self.ncliques = [len(cl) for cl in self.cliques]
        self.cliques_parent = symb.parent()
        self.localBusIdx = {}
        self.globalBusIdx_to_cliques = []
        self.N = 0
        for i in range(self.n):
            self.globalBusIdx_to_cliques.append([])
        for clique_idx,clique in enumerate(self.cliques):
            self.N+=len(clique)
            for local_idx,global_idx in enumerate(clique):
                self.localBusIdx[clique_idx,global_idx] = local_idx
                self.globalBusIdx_to_cliques[global_idx].append(clique_idx)
                
        self.cliques_intersection = []
        for clique_idx,clique in enumerate(self.cliques):
            if self.cliques_parent[clique_idx]==clique_idx:
                self.cliques_intersection.append([])
            else:
                set_clique,set_clique_parent = set(clique),set(self.cliques[self.cliques_parent[clique_idx]])
                self.cliques_intersection.append(set_clique.intersection(set_clique_parent))
        for i in range(len(self.cliques_intersection)):
            inter = list(self.cliques_intersection[i])
            inter.sort()
            self.cliques_intersection[i] = inter
        self.edges_to_clique = {}
        for (i,j) in self.edges:
            si = set(self.globalBusIdx_to_cliques[i])
            sj = set(self.globalBusIdx_to_cliques[j])
            inter = si.intersection(sj)
            assert(len(inter)>0)
            random.seed(i*self.n+j)
            self.edges_to_clique[(i,j)] = random.choice([i for i in inter])
        self.cliques_nbr = len(self.cliques)
                
    
    def test_thresholding(self,threshold):
        G=nx.Graph()
        G.add_nodes_from([i for i in range(self.cliques_nbr)])
        liste = []
        for clique_idx in range(self.cliques_nbr):
            if len(self.cliques_intersection[clique_idx])>threshold:
                liste.append((clique_idx,self.cliques_parent[clique_idx]))
        G.add_edges_from(liste)
        connected_components = nx.connected_components(G)
        
        size_list = []
        components = []
        for a in connected_components:
            node_set = set()
            for idx_cl in a:
                node_set = node_set.union(self.cliques[idx_cl])
            size_list.append(len(node_set))
            components.append(a)
        print("Taille du core:",max(size_list))
        core = components[np.argmax(np.array(size_list))]
                
        liste2 = []
        G2=nx.Graph()
        G2.add_nodes_from([i for i in range(self.cliques_nbr)])
        for clique_idx in range(self.cliques_nbr):
            liste2.append((clique_idx,self.cliques_parent[clique_idx]))
        G2.add_edges_from(liste2)
        G2.remove_nodes_from(core)
        connected_components2 = nx.connected_components(G2)
        size_list2 = []
        for a in connected_components2:
            node_set = set()
            for idx_cl in a:
                node_set = node_set.union(self.cliques[idx_cl])
            size_list2.append(len(node_set))
        print(len(size_list2))
        size_list2.sort()
        print(size_list2)
            
            
    def build_gamma_curvature(self):
        Qminbus, Qmaxbus = np.zeros(self.n), np.zeros(self.n)
        for idx_gen,gen in enumerate(self.genlist):
            bus,index = self.genlist[idx_gen]
            index_bus =  self.buslistinv[bus]
            Qminbus[index_bus]+=self.Qmin[idx_gen]
            Qmaxbus[index_bus]+=self.Qmax[idx_gen]
        self.gamma_curvature = Qmaxbus - Qminbus
        for i in range(self.n):
            assert(self.gamma_curvature[i]>=-1E-6)
            
   
    
    def test_status(self):
        """Check wether the lines are indeed active """
        for l in self.Yff:
            assert(self.status[l] ==1.0)
        for l in self.Yft:
            assert(self.status[l] ==1.0)
        for l in self.Ytf:
            assert(self.status[l] ==1.0)
        for l in self.Ytt:
            assert(self.status[l] ==1.0)
            
           
    def preprocessing_power_bounds(self):
        """Handle absence of bounds. """
        self.blocked_beta_gen_moins,self.blocked_beta_gen_plus, self.blocked_gamma_gen_moins, self.blocked_gamma_gen_plus = [],[],[],[]
        for i,gen in enumerate(self.genlist):
            assert(self.genlist[i]==gen)
            if self.Pmin[i]==-np.inf:
                print("Pmin = -inf for gen {0}. replaced by large negative value".format(gen))
                self.Pmin[i] = -myinf_power_lim
                self.blocked_beta_gen_moins.append(gen[0])
            if self.Pmax[i]==np.inf:
                print("Pmax = +inf for gen {0}. replaced by large positive value".format(gen))
                self.Pmax[i] = myinf_power_lim
                self.blocked_beta_gen_plus.append(gen[0])
            if self.Qmin[i]==-np.inf:
                print("Qmin = -inf for gen {0}. replaced by large negative value".format(gen))
                self.Qmin[i] = -myinf_power_lim
                self.blocked_gamma_gen_moins.append(gen[0])
            if self.Qmax[i]==np.inf:
                print("Qmax = +inf for gen {0}. replaced by large positive value".format(gen))
                self.Qmax[i] = myinf_power_lim
                self.blocked_gamma_gen_plus.append(gen[0])