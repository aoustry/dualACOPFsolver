# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 12:44:33 2021

@author: aoust
"""


import time, itertools, operator,osqp
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs,spsolve
from scipy.sparse import   coo_matrix,  identity, csc_matrix, hstack,vstack, diags
from tools import argmin_cumsum,gershgorin_bounds
from fractions import Fraction
###################Fixed parameters####################################
#My zeros
myEpsgradient = myZeroforCosts = 1E-6
my_zero_for_dual_variables = 1E-6
marginEVfroeb = 1E-5
magnitude_init_perturb = 1E-5

######################################################################


class dualACOPFsolver():
    
    def __init__(self, ACOPF):
        """
        
        Parameters
        ----------
        ACOPF : ACOPF instance (cf instance.py)
        
        config: dict

        -------
        Load the model.

        """
        
        self.name = ACOPF.name
        
        #Sizes
        self.baseMVA = ACOPF.baseMVA
        self.n, self.m, self.gn = ACOPF.n, ACOPF.m, ACOPF.gn
        self.N = ACOPF.N
        
        #Generator quantities
        self.C = ACOPF.C
        self.offset = ACOPF.offset
        self.lincost = ACOPF.lincost
        self.quadcost = ACOPF.quadcost
        self.genlist = ACOPF.genlist
        assert(len(self.genlist)==self.gn)
        self.Pmin, self.Qmin, self.Pmax, self.Qmax = ACOPF.Pmin, ACOPF.Qmin, ACOPF.Pmax, ACOPF.Qmax
                
        #Bus quantities
        self.buslist = ACOPF.buslist
        self.buslistinv = ACOPF.buslistinv
        for i in range(self.n):
            assert(self.buslistinv[self.buslist[i]]==i)
        self.busType = ACOPF.busType
        self.Vmin, self.Vmax = ACOPF.Vmin, ACOPF.Vmax
        self.A = ACOPF.A
        self.Pload, self.Qload = ACOPF.Pload, ACOPF.Qload
        
        #Lines quantities
        self.status = ACOPF.status
        self.cl = ACOPF.cl
        self.clinelist, self.clinelistinv = ACOPF.clinelist, ACOPF.clinelistinv
        self.Imax = ACOPF.Imax
        # self.scaling_lambda_f = [self.config["scaling_lambda"] for line in self.clinelistinv]
        # self.scaling_lambda_t = [self.config["scaling_lambda"] for line in self.clinelistinv]
        
        #Cliques quantities
        self.cliques_nbr = ACOPF.cliques_nbr
        self.cliques, self.ncliques = ACOPF.cliques, ACOPF.ncliques
        self.cliques_parent, self.cliques_intersection = ACOPF.cliques_parent, ACOPF.cliques_intersection 
        self.localBusIdx = ACOPF.localBusIdx
        self.rho = ACOPF.SVM
                
        #Lines quantities
        self.HM, self.ZM = ACOPF.HM, ACOPF.ZM
        self.assigned_lines, self.assigned_buses = ACOPF.assigned_lines, ACOPF.assigned_buses
        self.Nf, self.Nt = ACOPF.Nf, ACOPF.Nt
               
        #Build T and S matrices
        self.S,self.T = {},{}
        eta_counter = 0
        self.positive_eta, self.negative_eta = {}, {}
        for clique_idx in range(self.cliques_nbr):
            self.positive_eta[clique_idx] = []
            self.negative_eta[clique_idx] = []
            self.S[clique_idx], self.T[clique_idx] = [],[]
        for clique_idx in range(self.cliques_nbr):
            nc = self.ncliques[clique_idx]
            clique_father_idx = self.cliques_parent[clique_idx]
            nc_father = self.ncliques[clique_father_idx]
            for global_idx_bus_b in self.cliques_intersection[clique_idx]:
                assert(clique_father_idx!=clique_idx)
                local_index_bus_b = self.localBusIdx[clique_idx,global_idx_bus_b]
                self.T[clique_idx].append(coo_matrix(([1], ([local_index_bus_b],[local_index_bus_b])), shape = (nc,nc)).tocsc())
                local_index_bus_b_father = self.localBusIdx[clique_father_idx,global_idx_bus_b]
                self.S[clique_father_idx].append(coo_matrix(([1], ([local_index_bus_b_father],[local_index_bus_b_father])), shape = (nc_father,nc_father)).tocsc())
                self.positive_eta[clique_idx].append(eta_counter)
                self.negative_eta[clique_father_idx].append(eta_counter)
                eta_counter+=1
                
            for global_idx_bus_b,global_idx_bus_a in itertools.combinations(self.cliques_intersection[clique_idx], 2):
                assert(global_idx_bus_b<global_idx_bus_a)
                local_index_bus_b,local_index_bus_a = self.localBusIdx[clique_idx,global_idx_bus_b],self.localBusIdx[clique_idx,global_idx_bus_a]
                assert(local_index_bus_b!=local_index_bus_a)
                ref = coo_matrix(([1], ([local_index_bus_b],[local_index_bus_a])), shape = (nc,nc)).tocsc()
                local_index_bus_b_father,local_index_bus_a_father = self.localBusIdx[clique_father_idx,global_idx_bus_b],self.localBusIdx[clique_father_idx,global_idx_bus_a]
                assert(local_index_bus_b_father!=local_index_bus_a_father)
                ref_father = coo_matrix(([1], ([local_index_bus_b_father],[local_index_bus_a_father])), shape = (nc_father,nc_father)).tocsc()
                self.T[clique_idx].append(0.5*(ref+ref.H))
                self.T[clique_idx].append(0.5j*(ref-ref.H))
                self.S[clique_father_idx].append(0.5*(ref_father+ref_father.H))
                self.S[clique_father_idx].append(0.5j*(ref_father-ref_father.H))
                self.positive_eta[clique_idx].append(eta_counter)
                self.negative_eta[clique_father_idx].append(eta_counter)
                eta_counter+=1
                self.positive_eta[clique_idx].append(eta_counter)
                self.negative_eta[clique_father_idx].append(eta_counter)
                eta_counter+=1
        self.eta_nbr = eta_counter
        del eta_counter   
        for clique_idx in range(self.cliques_nbr):
            assert(len(self.S[clique_idx]) == len(self.negative_eta[clique_idx]) )
            assert(len(self.T[clique_idx]) == len(self.positive_eta[clique_idx]) )
        #Testing
        for code in self.S:
            
            for matrix in self.S[code]:
                assert(np.linalg.norm((matrix-matrix.H).toarray())<1E-7)
            for matrix in self.T[code]:
                assert(np.linalg.norm((matrix-matrix.H).toarray())<1E-7)
        
        self.__compute_capacities()
        self.__compute_matrix_operators()
        

        self.initial_values_set = False


    """Precomputations """                 
    def __compute_matrix_operators(self):
        self.dual_matrix_rows = {}
        self.dual_matrix_cols = {}
        self.MO = {}
        self.MO_transpose = {}
        self.vars = {}
        clique_sizes_offset = 0
        for idx_clique in range(self.cliques_nbr):
            dico_pairs_to_indexes = {}
            self.dual_matrix_rows[idx_clique] = []
            self.dual_matrix_cols[idx_clique] = []
            self.vars[idx_clique] = []
            Trows, Tcols, Tdata = [],[],[]
            counter = 0
            nc = self.ncliques[idx_clique]
            kc = len(self.assigned_buses[idx_clique])
            lc = len(self.assigned_lines[idx_clique])
            #Alpha coefficients
            for local_idx_bus in range(nc):
                dico_pairs_to_indexes[(local_idx_bus,local_idx_bus)] = counter
                self.dual_matrix_rows[idx_clique].append(local_idx_bus)
                self.dual_matrix_cols[idx_clique].append(local_idx_bus)
                Trows.append(counter)
                Tcols.append(local_idx_bus)
                Tdata.append(1)
                counter+=1
                self.vars[idx_clique].append(clique_sizes_offset+local_idx_bus)
            clique_sizes_offset+=nc
            #Beta coefficients
            for id_assignment,global_idx_bus in enumerate(list(self.assigned_buses[idx_clique])):
                self.vars[idx_clique].append(self.N+global_idx_bus)
                i_list,j_list = self.HM[idx_clique,global_idx_bus].nonzero()
                for aux in range(len(i_list)):
                    i,j = i_list[aux],j_list[aux]
                    if (i,j) in dico_pairs_to_indexes:
                        index_pair = dico_pairs_to_indexes[(i,j)]
                    else:
                        dico_pairs_to_indexes[(i,j)] = counter
                        self.dual_matrix_rows[idx_clique].append(i)
                        self.dual_matrix_cols[idx_clique].append(j)
                        index_pair = counter
                        counter+=1
                    Trows.append(index_pair)
                    Tcols.append(nc+id_assignment)
                    Tdata.append(self.HM[idx_clique,global_idx_bus][i,j])
                
            #Gamma coefficients
            for id_assignment,global_idx_bus in enumerate(list(self.assigned_buses[idx_clique])):
                i_list,j_list = self.ZM[idx_clique,global_idx_bus].nonzero()
                self.vars[idx_clique].append(self.N+self.n+global_idx_bus)
                for aux in range(len(i_list)):
                    i,j = i_list[aux],j_list[aux]
                    if (i,j) in dico_pairs_to_indexes:
                        index_pair = dico_pairs_to_indexes[(i,j)]
                    else:
                        dico_pairs_to_indexes[(i,j)] = counter
                        self.dual_matrix_rows[idx_clique].append(i)
                        self.dual_matrix_cols[idx_clique].append(j)
                        index_pair = counter
                        counter+=1
                    Trows.append(index_pair)
                    Tcols.append(nc+kc+id_assignment)
                    Tdata.append(1j*self.ZM[idx_clique,global_idx_bus][i,j])
                    
            #Lambda_f coefficients
            for id_assignment,global_idx_line in enumerate(list(self.assigned_lines[idx_clique])):
                i_list,j_list = self.Nf[idx_clique,global_idx_line].nonzero()
                self.vars[idx_clique].append(self.N+2*self.n+global_idx_line)
                for aux in range(len(i_list)):
                    i,j = i_list[aux],j_list[aux]
                    if (i,j) in dico_pairs_to_indexes:
                        index_pair = dico_pairs_to_indexes[(i,j)]
                    else:
                        dico_pairs_to_indexes[(i,j)] = counter
                        self.dual_matrix_rows[idx_clique].append(i)
                        self.dual_matrix_cols[idx_clique].append(j)
                        index_pair = counter
                        counter+=1
                    Trows.append(index_pair)
                    Tcols.append(nc+2*kc+id_assignment)
                    Tdata.append(self.Nf[idx_clique,global_idx_line][i,j]/self.scaling_lambda_f[global_idx_line]) 
                    
            #Lambda_t coefficients
            for id_assignment,global_idx_line in enumerate(list(self.assigned_lines[idx_clique])):
                i_list,j_list = self.Nt[idx_clique,global_idx_line].nonzero()
                self.vars[idx_clique].append(self.N+2*self.n+self.cl+global_idx_line)
                for aux in range(len(i_list)):
                    i,j = i_list[aux],j_list[aux]
                    if (i,j) in dico_pairs_to_indexes:
                        index_pair = dico_pairs_to_indexes[(i,j)]
                    else:
                        dico_pairs_to_indexes[(i,j)] = counter
                        self.dual_matrix_rows[idx_clique].append(i)
                        self.dual_matrix_cols[idx_clique].append(j)
                        index_pair = counter
                        counter+=1
                    Trows.append(index_pair)
                    Tcols.append(nc+2*kc+lc+id_assignment)
                    Tdata.append(self.Nt[idx_clique,global_idx_line][i,j]/self.scaling_lambda_t[global_idx_line])     
            
            #eta coefficients related to T
            assert(len(self.T[idx_clique]) == len(self.positive_eta[idx_clique]))
            for id_assignment, global_eta_id in enumerate(self.positive_eta[idx_clique]):
                i_list,j_list = self.T[idx_clique][id_assignment].nonzero()
                self.vars[idx_clique].append(self.N+2*self.n+2*self.cl+global_eta_id)
                for aux in range(len(i_list)):
                    i,j = i_list[aux],j_list[aux]
                    if (i,j) in dico_pairs_to_indexes:
                        index_pair = dico_pairs_to_indexes[(i,j)]
                    else:
                        dico_pairs_to_indexes[(i,j)] = counter
                        self.dual_matrix_rows[idx_clique].append(i)
                        self.dual_matrix_cols[idx_clique].append(j)
                        index_pair = counter
                        counter+=1
                    Trows.append(index_pair)
                    Tcols.append(nc+2*kc+2*lc+id_assignment)
                    Tdata.append(self.T[idx_clique][id_assignment][i,j])
                    
            #eta coefficients related to S
            lenPc = len(self.positive_eta[idx_clique])
            for id_assignment, global_eta_id in enumerate(self.negative_eta[idx_clique]):
                i_list,j_list = self.S[idx_clique][id_assignment].nonzero()
                self.vars[idx_clique].append(self.N+2*self.n+2*self.cl+global_eta_id)
                for aux in range(len(i_list)):
                    i,j = i_list[aux],j_list[aux]
                    if (i,j) in dico_pairs_to_indexes:
                        index_pair = dico_pairs_to_indexes[(i,j)]
                    else:
                        dico_pairs_to_indexes[(i,j)] = counter
                        self.dual_matrix_rows[idx_clique].append(i)
                        self.dual_matrix_cols[idx_clique].append(j)
                        index_pair = counter
                        counter+=1
                    Trows.append(index_pair)
                    Tcols.append(nc+2*kc+2*lc+lenPc+id_assignment)
                    Tdata.append(-self.S[idx_clique][id_assignment][i,j])
                    
            nvarc = nc+2*kc+2*lc+lenPc + len(self.negative_eta[idx_clique])
            assert(nvarc == len(self.vars[idx_clique]))
            self.MO[idx_clique] = coo_matrix((Tdata,(Trows,Tcols)),shape = (len(self.dual_matrix_rows[idx_clique]),nvarc)).tocsc()
            self.MO_transpose[idx_clique] = coo_matrix((Tdata,(Tcols,Trows)),shape = (nvarc,len(self.dual_matrix_rows[idx_clique]))).tocsc()
            
            self.vars[idx_clique] = np.array(self.vars[idx_clique])
        self.d = self.N+2*self.n+2*self.cl+self.eta_nbr
        cl = np.concatenate([idx_clique*np.ones(len(self.vars[idx_clique])) for idx_clique in range(self.cliques_nbr)])
        global_vars = np.concatenate([self.vars[idx_clique] for idx_clique in range(self.cliques_nbr)])
        self.clique_to_vars_matrix = coo_matrix((np.ones(len(cl)),(global_vars,cl)),shape = (self.d,self.cliques_nbr)).tocsc()
        
        self.dual_matrix_rows_inv = []
        for idx_cl in range(self.cliques_nbr):
            self.dual_matrix_rows_inv.append([])
            nc = self.ncliques[idx_cl]
            for i in range(nc):
                 self.dual_matrix_rows_inv[idx_cl].append([])
            for counter in range(len(self.dual_matrix_rows[idx_cl])):
                 self.dual_matrix_rows_inv[idx_cl][self.dual_matrix_rows[idx_cl][counter]].append(counter)
      
  
    def __compute_capacities(self):
        self.sumQmin,self.sumQmax = {},{}
        #Contribution of buses for beta and gamma
        for i in range(self.n):
            self.sumQmin[self.buslist[i]] = 0
            self.sumQmax[self.buslist[i]] = 0
        #Contribution of gen. for beta and gamma
        for idx_gen,gen in enumerate(self.genlist):
            bus,index = self.genlist[idx_gen]
            self.sumQmin[bus]+=self.Qmin[idx_gen]
            self.sumQmax[bus]+=self.Qmax[idx_gen]
            
    def __estimation(self):
        return np.mean(self.lincost) * sum(self.Pload)
    
    def __compute_upper_bound(self):
        ub = self.offset
        for idx_gen in range(self.gn):
            ub+=self.lincost[idx_gen]*self.Pmax[idx_gen] + self.quadcost[idx_gen]*self.Pmax[idx_gen]**2
        return ub
    
    def _soft_plus(self,x):
        if abs(x)>100:
            return max(x,0)
        return np.log(1+np.exp(x))
    
    def _soft_plus_epsilon(self,x,epsilon):
        return self._soft_plus(x/epsilon)*epsilon
    
    def sigmoid(self,x,epsilon):
        if (x/epsilon)<-100:
            return 0.0
        if (x/epsilon)>100:
            return 1.0
        return 1/(1+np.exp(-x/epsilon))
    
    def der_sigmoid_epsilon(self,x,epsilon):
        if (x/epsilon)<-100:
            return 0.0
        if (x/epsilon)>100:
            return 0.0
        s = self.sigmoid(x,epsilon)
        return (1/epsilon)*s*(1-s)
    
    """Oracles """          
            
    def __G_value_oracle(self,alpha,beta,gamma, lambda_f,lambda_t):
        """Return the value of function G at the current solution. """
        value = 0
        offset = 0
        for idx_clique in range(self.cliques_nbr):
            clique = self.cliques[idx_clique]
            for global_indx in clique:
                local_index = self.localBusIdx[idx_clique,global_indx]
                if (alpha[offset+local_index]<0):
                    value+= -alpha[offset+local_index]*(self.Vmin[global_indx]**2)
                else:
                    value+= -alpha[offset+local_index]*(self.Vmax[global_indx]**2)
            offset+=len(clique)
        
        #Contribution of buses for beta and gamma
        for i in range(self.n):
            value+= self.Pload[i] * beta[i]
            value+= self.Qload[i] * gamma[i]
        #Contribution of gen. for beta and gamma
        for idx_gen,gen in enumerate(self.genlist):
            bus,index = self.genlist[idx_gen]
            index_bus =  self.buslistinv[bus]
            #Contribution beta
            gap = beta[index_bus] - self.lincost[idx_gen]
            if abs(self.quadcost[idx_gen]) >= myZeroforCosts:
                if (gap<2*self.quadcost[idx_gen]*self.Pmin[idx_gen]):
                    value += self.Pmin[idx_gen] * (self.quadcost[idx_gen]*self.Pmin[idx_gen] - gap)
                elif (gap> 2*self.quadcost[idx_gen]*self.Pmax[idx_gen]):
                    value += self.Pmax[idx_gen] * (self.quadcost[idx_gen]*self.Pmax[idx_gen] - gap)
                else:
                    value += - (gap**2)/(4*self.quadcost[idx_gen])
            else:
                if (gap<0):
                    value += - gap * self.Pmin[idx_gen]
                else:
                    value += - gap * self.Pmax[idx_gen]
            
            #Contribution gamma
            if (gamma[index_bus]<0):
                value += -gamma[index_bus] *self.Qmin[idx_gen]
            else:
                value += -gamma[index_bus] *self.Qmax[idx_gen]
                
        #Contribution lambda
        for idx_line in range(self.cl):
            if lambda_f[idx_line]>0:
                value += -lambda_f[idx_line]*(self.Imax[idx_line]**2)/self.scaling_lambda_f[idx_line]
            if lambda_t[idx_line]>0:
                value += -lambda_t[idx_line]*(self.Imax[idx_line]**2)/self.scaling_lambda_t[idx_line]
        
        return value + self.offset
    
    
    def __G_value_oracle_smoothed(self,alpha,beta,gamma, lambda_f,lambda_t,epsilon):
        """Return the value of function G at the current solution. """
        value = 0
        index_offset = 0
        for idx_clique in range(self.cliques_nbr):
            clique = self.cliques[idx_clique]
            for global_indx in clique:
                local_index = self.localBusIdx[idx_clique,global_indx]
                value+= self._soft_plus_epsilon(-alpha[index_offset+local_index],epsilon)*(self.Vmin[global_indx]**2)
                value+= -self._soft_plus_epsilon(alpha[index_offset+local_index],epsilon)*(self.Vmax[global_indx]**2)
            index_offset+=len(clique)
        
        #Contribution of buses for beta and gamma
        for i in range(self.n):
            value+= self.Pload[i] * beta[i]
            value+= self.Qload[i] * gamma[i]
        #Contribution of gen. for beta and gamma
        for idx_gen,gen in enumerate(self.genlist):
            bus,index = self.genlist[idx_gen]
            index_bus =  self.buslistinv[bus]
            #Contribution beta
            gap = beta[index_bus] - self.lincost[idx_gen]
            assert(self.quadcost[idx_gen] <= myZeroforCosts)
            value += self._soft_plus_epsilon(- gap,epsilon) * self.Pmin[idx_gen]
            value += - self._soft_plus_epsilon(gap,epsilon) * self.Pmax[idx_gen]
            
            #Contribution gamma
            value += self._soft_plus_epsilon(-gamma[index_bus],epsilon) *self.Qmin[idx_gen]
            value += -self._soft_plus_epsilon(gamma[index_bus],epsilon) *self.Qmax[idx_gen]
                
        #Contribution lambda
        assert(self.cl==0)
        #to be continued
        # for idx_line in range(self.cl):
        #     if lambda_f[idx_line]>0:
        #         value += -lambda_f[idx_line]*(self.Imax[idx_line]**2)/self.scaling_lambda_f[idx_line]
        #     if lambda_t[idx_line]>0:
        #         value += -lambda_t[idx_line]*(self.Imax[idx_line]**2)/self.scaling_lambda_t[idx_line]
        
        return value + self.offset
    
    
    def __G_gradient_oracle_smoothed(self,alpha,beta,gamma, lambda_f,lambda_t,epsilon):
        """Return the gradient of the smoothed function G at the current solution. """
        index_offset = 0
        grad_alpha = np.zeros(len(alpha))
        for idx_clique in range(self.cliques_nbr):
            clique = self.cliques[idx_clique]
            for global_indx in clique:
                local_index = self.localBusIdx[idx_clique,global_indx]
                grad_alpha[index_offset+local_index]+= -self.sigmoid(-alpha[index_offset+local_index],epsilon)*(self.Vmin[global_indx]**2)
                grad_alpha[index_offset+local_index]+= -self.sigmoid(alpha[index_offset+local_index],epsilon)*(self.Vmax[global_indx]**2)
            index_offset+=len(clique)
        
        grad_beta = np.zeros(self.n)
        grad_gamma = np.zeros(self.n)
        #Contribution of buses for beta and gamma
        for i in range(self.n):
            grad_beta[i]+= self.Pload[i] 
            grad_gamma[i]+= self.Qload[i] 
        #Contribution of gen. for beta and gamma
        for idx_gen,gen in enumerate(self.genlist):
            bus,index = self.genlist[idx_gen]
            index_bus =  self.buslistinv[bus]
            #Contribution beta
            gap = beta[index_bus] - self.lincost[idx_gen]
            assert(self.quadcost[idx_gen] <= myZeroforCosts)
            grad_beta[index_bus] += -self.sigmoid(- gap,epsilon) * self.Pmin[idx_gen]
            grad_beta[index_bus] += -self.sigmoid(gap,epsilon) * self.Pmax[idx_gen]
            
            #Contribution gamma
            grad_gamma[index_bus] += -self.sigmoid(-gamma[index_bus],epsilon) *self.Qmin[idx_gen]
            grad_gamma[index_bus] += -self.sigmoid(gamma[index_bus],epsilon) *self.Qmax[idx_gen]
                
        #Contribution lambda
        assert(self.cl==0)
        #to be continued
        # for idx_line in range(self.cl):
        #     if lambda_f[idx_line]>0:
        #         value += -lambda_f[idx_line]*(self.Imax[idx_line]**2)/self.scaling_lambda_f[idx_line]
        #     if lambda_t[idx_line]>0:
        #         value += -lambda_t[idx_line]*(self.Imax[idx_line]**2)/self.scaling_lambda_t[idx_line]
        
        return np.concatenate([grad_alpha, grad_beta, grad_gamma])
    
    def __G_hessian_oracle_smoothed(self,alpha,beta,gamma, lambda_f,lambda_t,epsilon):
        """Return the hessian diagonal of the smoothed function G at the current solution. """
        index_offset = 0
        hess_alpha = np.zeros(len(alpha))
        for idx_clique in range(self.cliques_nbr):
            clique = self.cliques[idx_clique]
            for global_indx in clique:
                local_index = self.localBusIdx[idx_clique,global_indx]
                hess_alpha[index_offset+local_index]+= self.der_sigmoid_epsilon(-alpha[index_offset+local_index],epsilon)*(self.Vmin[global_indx]**2)
                hess_alpha[index_offset+local_index]+= -self.der_sigmoid_epsilon(alpha[index_offset+local_index],epsilon)*(self.Vmax[global_indx]**2)
            index_offset+=len(clique)
        
        hess_beta = np.zeros(self.n)
        hess_gamma = np.zeros(self.n)
        #Contribution of gen. for beta and gamma
        for idx_gen,gen in enumerate(self.genlist):
            bus,index = self.genlist[idx_gen]
            index_bus =  self.buslistinv[bus]
            #Contribution beta
            gap = beta[index_bus] - self.lincost[idx_gen]
            assert(self.quadcost[idx_gen] <= myZeroforCosts)
            hess_beta[index_bus] += self.der_sigmoid_epsilon(- gap,epsilon) * self.Pmin[idx_gen]
            hess_beta[index_bus] += -self.der_sigmoid_epsilon(gap,epsilon) * self.Pmax[idx_gen]
            
            #Contribution gamma
            hess_gamma[index_bus] += self.der_sigmoid_epsilon(-gamma[index_bus],epsilon) *self.Qmin[idx_gen]
            hess_gamma[index_bus] += -self.der_sigmoid_epsilon(gamma[index_bus],epsilon) *self.Qmax[idx_gen]
                
        #Contribution lambda
        assert(self.cl==0)
        #to be continued
        # for idx_line in range(self.cl):
        #     if lambda_f[idx_line]>0:
        #         value += -lambda_f[idx_line]*(self.Imax[idx_line]**2)/self.scaling_lambda_f[idx_line]
        #     if lambda_t[idx_line]>0:
        #         value += -lambda_t[idx_line]*(self.Imax[idx_line]**2)/self.scaling_lambda_t[idx_line]
        
        return np.concatenate([hess_alpha, hess_beta, hess_gamma])
        
    
    
    def test(self):
        
        for i in range(4):
            index_offset = 0
            for idx_clique in range(self.cliques_nbr):
                index_offset+=len(self.cliques[idx_clique])
            alpha = 4*(2*np.random.rand(index_offset)-1)
            beta,gamma = 4*(2*np.random.rand(self.n)-1),4*(2*np.random.rand(self.n)-1)
            lambda_f,lambda_t = np.zeros(self.cl),np.zeros(self.cl)
            eta = 4*(2*np.random.rand(self.eta_nbr)-1)
            epsilon = 1
            print(len(alpha),len(beta),len(gamma),len(eta))
            n = 1e-8
            t = time.time()
            value, gradient, sparse_hessian = self.value_smoothed_with_derivatives(alpha, beta, gamma, lambda_f, lambda_t, eta,epsilon)
            print('Temps = {0}'.format(time.time()-t))
            v,w= eigs(sparse_hessian,k=6,sigma = 0)
            #print(sparse_hessian)
            print((v))
            
            # for i in range(5):
            #     d_alpha = 2*np.random.rand(index_offset)-1
            #     d_beta,d_gamma = 2*np.random.rand(self.n)-1,2*np.random.rand(self.n)-1
            #     d_eta = 2*np.random.rand(self.eta_nbr)-1
            #     d_lambda_f = d_lambda_t = np.zeros(0)
            #     delta = np.concatenate([d_alpha,d_beta,d_gamma,d_lambda_f , d_lambda_t,d_eta])
            #     value2, gradient2, sparse_hessian2 = self.value_smoothed_with_derivatives(alpha+n*d_alpha, beta+n*d_beta, gamma+n*d_gamma, lambda_f+n*d_lambda_f, lambda_t+n*d_lambda_t, eta+n*d_eta,epsilon)
            #     print(np.linalg.norm(((value2-value)/n)-gradient.dot(delta)),abs(gradient.dot(delta)))
            #     print(np.linalg.norm(((gradient2-gradient)/n)-sparse_hessian.dot(delta)),np.linalg.norm(sparse_hessian.dot(delta)))

            
    def __matrix_operator(self,xc,idx_clique):
        """
        Parameters
        ----------
        xc : numpy array.
        idx_clique : int.
        Returns
        -------
        res : the dual matrix associated with clique c
        """
        vector_version = self.MO[idx_clique].dot(xc)
        nc = self.ncliques[idx_clique]
        return coo_matrix((vector_version, (self.dual_matrix_rows[idx_clique],self.dual_matrix_cols[idx_clique])), shape = (nc,nc))
    
    def __SVD(self,xc,idx_clique):
        matrix = (self.__matrix_operator(xc,idx_clique)).toarray()
        s,U = np.linalg.eigh(matrix)
        return U, s
    
    def __spectral_sftplus(self,eigenvals,epsilon):
        return sum([self._soft_plus_epsilon(el, epsilon) for el in  eigenvals])
    
    def __spectral_grad_matrix(self,U,eigenvals,epsilon):
        der = np.array([self.sigmoid(el,epsilon) for el in eigenvals])
        boolean = np.absolute(der)>1e-9
        W = U[:,boolean]
        return  W.dot(diags(der[boolean]).dot(np.conj(W.T)))
    
    def __spectral_matrix_M(self,eigenvals,epsilon):
        xi_diag = np.diag([self.der_sigmoid_epsilon(el,epsilon) for el in eigenvals])
        nc = len(eigenvals)
        M = np.zeros((nc,nc)) + xi_diag
        m_row_ind,m_col_ind = [i for i in range(nc)],[i for i in range(nc)]
        for i in range(nc):
            for j in range(i+1,nc):
                if abs(eigenvals[i]-eigenvals[j])<1e-12:
                    M[i,j] = self.der_sigmoid_epsilon(eigenvals[i],epsilon)
                    M[j,i] = M[i,j]
                else:
                    M[i,j] = (self.sigmoid(eigenvals[i],epsilon)-self.sigmoid(eigenvals[j],epsilon))/(eigenvals[i]-eigenvals[j])
                    M[j,i] = M[i,j]
                if abs(M[i,j])>1e-7:
                    m_row_ind.append(i)
                    m_col_ind.append(j)
                    m_row_ind.append(j)
                    m_col_ind.append(i)
        return M,m_row_ind,m_col_ind

    def __Atilde(self, idx_clique,m_row_ind, m_col_ind , U):
        assert(U.shape[0]==self.ncliques[idx_clique])
        assert(len(m_row_ind)==len(m_col_ind))
        X1 = np.conj(U[:,m_row_ind][self.dual_matrix_rows[idx_clique],:]) 
        X2 = (U[:,m_col_ind][self.dual_matrix_cols[idx_clique],:])
        P = X1*X2
        return ((self.MO_transpose[idx_clique]).dot(P)).T
    
    def __spectral_grad(self,idx_clique,U,eigenvals,epsilon):
        grad_mat_m = self.__spectral_grad_matrix(U,eigenvals,epsilon)
        grad_mat_m_vec = grad_mat_m[self.dual_matrix_rows[idx_clique],self.dual_matrix_cols[idx_clique]]
        return np.real((np.conj((self.MO_transpose[idx_clique])).dot(grad_mat_m_vec)).T)
    
    def __spectral_hessian(self,idx_clique,U,eigenvals,epsilon):
        M, m_row_ind,m_col_ind = self.__spectral_matrix_M(eigenvals,epsilon)
        D = diags(M[m_row_ind,m_col_ind])
        if len(m_row_ind)==0:
            return False,0
        At = self.__Atilde(idx_clique,m_row_ind, m_col_ind , U)
        return True,np.real( (np.conj(At.T)).dot(D.dot(At)))
        
    """External routines """
    
    def value(self,alpha, beta, gamma, lambda_f, lambda_t, eta):
        """Function to evaluate a dual solution. No side effect on the class attributes. """
        Fval = 0
        theta = np.concatenate([alpha, beta, gamma, lambda_f, lambda_t, eta])
        for idx_clique in range(self.cliques_nbr):
            U,s = self.__SVD(theta[self.vars[idx_clique]],idx_clique)
            Fval+= self.rho[idx_clique]* min(0,s.min()) 
        Gval = self.__G_value_oracle(alpha, beta, gamma, lambda_f, lambda_t)
        return Gval + Fval
    
    def value_smoothed(self,alpha, beta, gamma, lambda_f, lambda_t, eta,epsilon):
        """Function to evaluate a dual solution. No side effect on the class attributes. """
        Fval = 0
        theta = np.concatenate([alpha, beta, gamma, lambda_f, lambda_t, eta])
        for idx_clique in range(self.cliques_nbr):
            U,s = self.__SVD(theta[self.vars[idx_clique]],idx_clique)
            Fval+= -self.rho[idx_clique]* self.__spectral_sftplus(-s,epsilon)
        Gval = self.__G_value_oracle_smoothed(alpha, beta, gamma, lambda_f, lambda_t,epsilon)
        return Gval + Fval
    
    def value_smoothed_with_derivatives(self,alpha, beta, gamma, lambda_f, lambda_t, eta,epsilon):
        """Function to evaluate dual smooth function and derivatives """
        theta = np.concatenate([alpha, beta, gamma, lambda_f, lambda_t, eta])
        value = self.__G_value_oracle_smoothed(alpha, beta, gamma, lambda_f, lambda_t,epsilon)
        gradient = np.zeros(len(theta))
        gradient_g = self.__G_gradient_oracle_smoothed(alpha,beta,gamma, lambda_f,lambda_t,epsilon)
        gradient[:len(gradient_g)] = gradient_g
        hessian_g = self.__G_hessian_oracle_smoothed(alpha,beta,gamma, lambda_f,lambda_t,epsilon)
        full_hessian = {(i,i):hessian_g[i] for i in range(len(hessian_g))}
        for idx_clique in range(self.cliques_nbr):
            U,s = self.__SVD(theta[self.vars[idx_clique]],idx_clique)
            value+= -self.rho[idx_clique]* self.__spectral_sftplus(-s,epsilon)
            gradient[self.vars[idx_clique]]+= self.rho[idx_clique]*self.__spectral_grad(idx_clique,U,-s,epsilon)
            boolean,H =  self.__spectral_hessian(idx_clique,U,-s,epsilon)
            H = -self.rho[idx_clique]*H
            if boolean:
                nc = len(self.vars[idx_clique])
                for aux1 in range(nc):
                    for aux2 in range(nc):
                        i,j = self.vars[idx_clique][aux1],self.vars[idx_clique][aux2]
                        if (i,j) in full_hessian:
                            full_hessian[(i,j)]+=H[aux1,aux2]
                        else:
                            full_hessian[(i,j)]=H[aux1,aux2]
        hdata,hrc = list(full_hessian.values()), [list(t) for t in zip(*(list(full_hessian.keys())))]
        sparse_hessian = coo_matrix((hdata,(hrc[0],hrc[1])),shape = (len(theta),len(theta))).tocsc()
        return value, gradient, sparse_hessian 
    
    def test_spectral_der(self,alpha, beta, gamma, lambda_f, lambda_t, eta,epsilon):
        """Function to evaluate dual smooth function and derivatives """
        theta = np.concatenate([alpha, beta, gamma, lambda_f, lambda_t, eta])
        dtheta = (2*np.random.rand(len(theta))-1)
        l = 1e-6
        theta_aux = theta + l*dtheta
        
        for idx_clique in range(self.cliques_nbr):
            nc = self.ncliques[idx_clique]
            
            U,s = self.__SVD(theta[self.vars[idx_clique]],idx_clique)
            val = self.__spectral_sftplus(s,epsilon)
            grad = self.__spectral_grad(idx_clique,U,s,epsilon)
            
            U_aux,s_aux = self.__SVD(theta_aux[self.vars[idx_clique]],idx_clique)
            val_aux = self.__spectral_sftplus(s_aux,epsilon)
            grad_aux = self.__spectral_grad(idx_clique,U_aux,s_aux,epsilon)
            
            # print("------{0}--------".format(self.ncliques[idx_clique]))
            # print(s)
            # print(abs(grad.dot(dtheta[self.vars[idx_clique]])-(val_aux-val)/l))
            # assert(np.linalg.norm(np.imag(grad))<1e-6)
            # t0 = time.time()
            boolean,Hess =  self.__spectral_hessian(idx_clique,U,s,epsilon)
            #print("Hessian error = {0}".format(np.linalg.norm(Hess.dot((dtheta[self.vars[idx_clique]]))-(grad_aux-grad)/l)))
            # H = (self.__matrix_operator(dtheta[self.vars[idx_clique]],idx_clique)).toarray()
            # Htilde = np.conj(U.T).dot(H).dot(U)
            # Mmat,m_row_ind,m_col_ind = self.__spectral_matrix_M(s,epsilon)
            # fd = np.conj(Htilde.reshape(nc*nc)).dot((Mmat*Htilde).reshape(nc*nc))
            # testHessianDotDelta = np.zeros(len(self.vars[idx_clique]))
            # for i in range(len(self.vars[idx_clique])):
            #     basis = np.zeros(len(self.vars[idx_clique]))
            #     basis[i]=1
            #     H2 = (self.__matrix_operator(basis,idx_clique)).toarray()
            #     Htilde2 = np.conj(U.T).dot(H2).dot(U)
            #     testHessianDotDelta[i] = np.conj(Htilde2.reshape(nc*nc)).dot((Mmat*Htilde).reshape(nc*nc))
            # print('testHessian = {0}'.format(np.linalg.norm(testHessianDotDelta-(grad_aux-grad)/l)))
            # '''TestHessian semble fonctionner. Cela signifie que la formule de Mmat est bonne mais problème dans l'opérateur spectral_hessian'''
            # fd2 = (Hess.dot(dtheta[self.vars[idx_clique]])).dot(dtheta[self.vars[idx_clique]])
            # print(fd,fd2)
            # print("minvp",min(np.linalg.eigvals(np.real(Hess))))
            # """ Il semblerait que fd soit bien calculé, mais pas la hessienne"""
            
            # At = self.__Atilde(idx_clique,m_row_ind, m_col_ind , U)
            # Htildetest = At.dot(dtheta[self.vars[idx_clique]])
            # print("Testhtilde = {0}".format(np.linalg.norm(Htildetest-Htilde[m_row_ind,m_col_ind])))
            # print('La fonction Atilde semble bien codée.')
    
    def certified_value(self,alpha, beta, gamma, lambda_f, lambda_t, eta):
        """Function to evaluate a dual solution, based on the SVD certificates. No side effect on the class attributes. """
        Fval = 0
        theta = np.concatenate([alpha, beta, gamma, lambda_f, lambda_t, eta])
        for idx_clique in range(self.cliques_nbr):
            U,s = self.__SVD(theta[self.vars[idx_clique]],idx_clique)
            matrix = (self.__matrix_operator(theta[self.vars[idx_clique]],idx_clique)).toarray()
            epsilon = matrix - (U).dot(np.diag(s)).dot(np.conj(U.T))
            shift,_ = gershgorin_bounds(epsilon)
            Fval+= self.rho[idx_clique]* min(0,s.min()+shift) 
        Gval = self.__G_value_oracle(alpha, beta, gamma, lambda_f, lambda_t)
        return Gval + Fval
    
    def solve(self,epsilon):
        """Init """
        alpha =   np.zeros(self.N)
        beta = np.zeros(self.n)
        gamma = np.zeros(self.n)
        lambda_f =  np.zeros(self.cl)
        lambda_t =  np.zeros(self.cl)
        eta =  np.zeros(self.eta_nbr)
        m = self.N+2*self.n+2*self.cl+self.eta_nbr
        a,b,c,d,e = self.N,self.N+self.n,self.N+2*self.n,self.N+2*self.n+self.cl,self.N++2*self.n+2*self.cl
        for i in range(10):
            
            theta = np.concatenate([alpha, beta, gamma, lambda_f, lambda_t, eta])
            value, gradient, sparse_hessian = self.value_smoothed_with_derivatives(alpha, beta, gamma, lambda_f, lambda_t, eta,epsilon)
            print(value)
            delta = spsolve(sparse_hessian,-gradient)
            theta = theta+delta
            alpha,beta, gamma, lambda_f,lambda_t, eta =  theta[:a],theta[a:b],theta[b:c],theta[c:d],theta[d:e],theta[e:]
            print('Implement line search')