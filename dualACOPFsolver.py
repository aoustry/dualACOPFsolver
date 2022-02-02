# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 12:44:33 2021

@author: aoust
"""


import time, itertools, operator,osqp
import numpy as np
from scipy.sparse import   coo_matrix,  identity, csc_matrix, hstack,vstack
from tools import argmin_cumsum,gershgorin_bounds
from fractions import Fraction
###################Fixed parameters####################################
#My zeros
myEpsgradient = myZeroforCosts = 1E-6
my_zero_for_dual_variables = 1E-6
marginEVfroeb = 1E-5
magnitude_init_perturb = 1E-5

#Proximal parameters
mu_LB, mu_UB = 1E-7, 100
hybridation_ratio_lb,hybridation_ratio_ub = 1E-5,1000
increase_ratio, decrease_ratio = 1.05, 0.95
nb_null_step_before_increase =5
low_error_prop, high_error_prop = 0.15,0.99
max_number_of_consecutive_nsteps = 50
ratio_stall_condition = 1E-6

#Bundle management parameters
ratio_added_cuts_end = 0.9
epsilon_eigencuts = 0.01
serious_steps_before_deletion = 5

#OSQP Parameters
osqp_eps_rel =1E-7
osqp_eps_dual =1E-5
osqp_polish=True
osqp_verbose = False
######################################################################


class dualACOPFsolver():
    
    def __init__(self, ACOPF, config):
        """
        
        Parameters
        ----------
        ACOPF : ACOPF instance (cf instance.py)
        
        config: dict

        -------
        Load the model.

        """
        
        self.name = ACOPF.name
        self.config = config
        
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
        self.scaling_lambda_f = [self.config["scaling_lambda"] for line in self.clinelistinv]
        self.scaling_lambda_t = [self.config["scaling_lambda"] for line in self.clinelistinv]
        
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
        
        #### Initialize cutting planes containers
        self.eigenvectors,self.eigencuts_coefs, self.eigencuts_idx,self.eigencuts_dual,self.eigencuts_slack, self.eigencuts_sstep = {},{},{},{},{},{}
        self.eigencuts_counter = 0
        
        self.initial_values_set = False

    """Logs and output stream """
    def __log(self):
        for logger in self.loggers:
            logger.log()
            
    def __final_status_log(self,text):
        with open("output/"+self.name+"_"+self.config['name']+"_"+self.loggers[0].date+"STATUS_BUNDLE.txt", 'w+') as txt_file:
            txt_file.write(text)
            txt_file.close()
         
        
            
    def __initial_info(self):
        if self.verbose:
            print("\n")
            print("""-----------------------------------------------------------------
                       dualACOPFsolver v1.0
                          Antoine Oustry\n  Laboratoire d'informatique de l'Ecole polytechnique (LIX), 2021\n-----------------------------------------------------------------""")
            print("Instance :         {0}".format(self.name))
            print("                   (current line constr: {0})".format(self.cl>0))
            
            print("Solver parameters: maxit = {0}, m = {1}".format(self.maxit,self.mbundle))
            print("                   rel_tol = {0}, abs_tol (->delta) = {1}".format(format(self.config['rel_tol'],".1e"), format(self.tol,".1e")))
            print("                   ratio_added_cuts = {0}, aggreg. = {1}".format(self.config['ratio_added_cuts'],self.config['aggregation']))
            print("                   warm_start = {0}".format(self.warmstart))
            print("It.   Best obj.  Delta      Grad.      Mu")
    
    def __info(self, terminated=False):
        
        if self.verbose and (self.it%50==0 or terminated):
            string_it = str(self.it) + " "*(6-len(str(self.it)))
            string_best_value_aux = ("%.7g" % self.best_certified_value) 
            string_best_value = string_best_value_aux + " "*(11-len(string_best_value_aux))
            string_delta = format(self.delta, ".1E") + (" "*(11-len(format(self.delta, ".1e"))))
            string_grad_norm = format(self.grad_norm, ".1E") + (" "*(11-len(format(self.grad_norm, ".1e"))))
            string_mu = format(self.kappa, ".1E") + (" "*(11-len(format(self.kappa, ".1e"))))
            
            print(string_it+string_best_value+string_delta+string_grad_norm+string_mu)
    
    def __stall_condition(self,values):
        if len(values)<max_number_of_consecutive_nsteps:
            return False
        start = values[len(values)-max_number_of_consecutive_nsteps]
        if start < 0:
            return False
        finish = values[-1]
        return (abs(finish-start)/abs(start))<ratio_stall_condition
    
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
                
    """Oracles """
    
    
    def __G_gradient_beta(self, beta_val):
        """Return a supergradient of G at the current solution w.r.t. beta variables. """
        grad = np.zeros(self.n)
        
        #Contribution of buses for beta 
        for i in range(self.n):
            grad[i] = self.Pload[i]
            
        #Contribution of gen. for beta 
        for idx_gen,gen in enumerate(self.genlist):
            bus,index = self.genlist[idx_gen]
            index_bus =  self.buslistinv[bus]
            #Contribution beta
            gap = beta_val[index_bus] - self.lincost[idx_gen]
            if abs(self.quadcost[idx_gen]) >= myZeroforCosts:
                if (gap<2*self.quadcost[idx_gen]*self.Pmin[idx_gen]):
                    grad[index_bus] += -self.Pmin[idx_gen]
                elif (gap> 2*self.quadcost[idx_gen]*self.Pmax[idx_gen]):
                    grad[index_bus] += -self.Pmax[idx_gen]
                else:
                    grad[index_bus] +=  - (gap)/(2*self.quadcost[idx_gen])
            else:
                
                if (gap<0):
                    grad[index_bus] += -self.Pmin[idx_gen]
                else:
                    grad[index_bus] += -self.Pmax[idx_gen]
            
        return grad
          
            
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
    
    def __G_value_beta(self,beta):
        """Return the value of function G at the current solution. """
        value = np.zeros(self.n)
        
        for i in range(self.n):
            value[i] = self.Pload[i] * beta[i]
            
        #Contribution of gen. for beta and gamma
        for idx_gen,gen in enumerate(self.genlist):
            bus,index = self.genlist[idx_gen]
            index_bus =  self.buslistinv[bus]
            #Contribution beta
            gap = beta[index_bus] - self.lincost[idx_gen]
            if abs(self.quadcost[idx_gen]) >= myZeroforCosts:
                if (gap<2*self.quadcost[idx_gen]*self.Pmin[idx_gen]):
                    value[index_bus] += self.Pmin[idx_gen] * (self.quadcost[idx_gen]*self.Pmin[idx_gen] - gap)
                elif (gap> 2*self.quadcost[idx_gen]*self.Pmax[idx_gen]):
                    value[index_bus] += self.Pmax[idx_gen] * (self.quadcost[idx_gen]*self.Pmax[idx_gen] - gap)
                else:
                    value[index_bus] += - (gap**2)/(4*self.quadcost[idx_gen])
            else:
                if (gap<0):
                    value[index_bus] += - gap * self.Pmin[idx_gen]
                else:
                    value[index_bus] += - gap * self.Pmax[idx_gen]
                
        return value 
        
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
    
    # def __matrix_operator_rationals(self,xc,idx_clique):
    #     """
    #     Parameters
    #     ----------
    #     xc : numpy array of fractions.
    #     idx_clique : int.
    #     Returns
    #     -------
    #     res : the dual matrix associated with clique c
    #     """
    #     for m in xc:
    #         assert(type(m)==Fraction)
        
    #     # self.MO[idx_clique]
        
    #     # vector_version = self.MO[idx_clique].dot(xc)
    #     # nc = self.ncliques[idx_clique]
    #     # return coo_matrix((vector_version, (self.dual_matrix_rows[idx_clique],self.dual_matrix_cols[idx_clique])), shape = (nc,nc))
             
    
    def __SVD(self,xc,idx_clique):
        matrix = (self.__matrix_operator(xc,idx_clique)).toarray()
        s,U = np.linalg.eigh(matrix)
        return U, s
    
    """Bundle management """
    def __add_betacut(self,grad_beta, value_function_beta,i):
        self.betacuts_coefs[self.betacuts_counter] = grad_beta
        self.betacuts_idx[self.betacuts_counter]=(i)
        self.betacuts_offset[self.betacuts_counter]=(value_function_beta - self.beta_val[i] * grad_beta)
        self.betacuts_dual[self.betacuts_counter] = 0
        self.betacuts_slack[self.betacuts_counter] = 0
        self.betacuts_counter+=1
    
    def __delete_betacuts(self):
        toremove = []
        for key in self.betacuts_coefs:
            if abs(self.betacuts_dual[key])<my_zero_for_dual_variables:
                toremove.append(key)
        for key in toremove:     
            self.betacuts_coefs.pop(key)
            self.betacuts_idx.pop(key)
            self.betacuts_offset.pop(key)
            self.betacuts_dual.pop(key)
            self.betacuts_slack.pop(key)
            
    def __add_eigencuts(self,U,s,idx_clique):
        mini = s.min()
        nc = self.ncliques[idx_clique]
        for i in range(nc):
            if s[i] <= mini + epsilon_eigencuts:
                vector = U[:,i]
                v1 = np.conj(vector[self.dual_matrix_rows[idx_clique]])
                v2 = vector[self.dual_matrix_cols[idx_clique]]
                coefs = np.real((self.MO_transpose[idx_clique]).dot(v1*v2))
                self.eigenvectors[self.eigencuts_counter] = vector
                self.eigencuts_coefs[self.eigencuts_counter] = coefs
                self.eigencuts_idx[self.eigencuts_counter] =(idx_clique)
                self.eigencuts_dual[self.eigencuts_counter] = 0
                self.eigencuts_slack[self.eigencuts_counter] = 0
                self.eigencuts_sstep[self.eigencuts_counter] = self.serious_step_number
                self.eigencuts_counter+=1      
    
    def __delete_eigencuts(self):
        toremove = []
        for key in self.eigencuts_coefs:
            if (self.eigencuts_sstep[key]<self.serious_step_number - 3) and abs(self.eigencuts_dual[key])<my_zero_for_dual_variables   :
                 toremove.append(key)
        for key in toremove:
            self.eigenvectors.pop(key)
            self.eigencuts_coefs.pop(key)
            self.eigencuts_idx.pop(key)
            self.eigencuts_dual.pop(key)
            self.eigencuts_slack.pop(key)
            self.eigencuts_sstep.pop(key)
    
    def __aggregate_eigencuts(self):
        cliques_to_keys = []
        for i in range(self.cliques_nbr):
            cliques_to_keys.append([])
        for key in self.eigencuts_idx:
            idx_cl = self.eigencuts_idx[key]
            cliques_to_keys[idx_cl].append(key)
        eigenvectors, eigencuts_coefs, eigencuts_idx, eigencuts_dual, eigencuts_slack,eigencuts_sstep = {},{},{},{},{},{}
        for idx_cl in range(self.cliques_nbr):
            vectorized_matrices = []
            weights = []
            nc = self.ncliques[idx_cl]
            if len(cliques_to_keys[idx_cl]):
                for key in cliques_to_keys[idx_cl]:
                    vector = self.eigenvectors[key].reshape((1,nc))
                    matrice = ((vector.T).dot(np.conj(vector)))
                    vectorized_matrices.append(matrice.flatten())
                    weights.append(self.eigencuts_dual[key])
                vectorized_matrices = np.array(vectorized_matrices)
                weights = np.array(weights)
                sum_matrix = weights.dot(vectorized_matrices)
                sum_matrix = sum_matrix.reshape((nc,nc))
                s,U = np.linalg.eigh(sum_matrix)
                
                for i in range(nc):
                    if s[i] >my_zero_for_dual_variables:
                        vector = U[:,i]
                        v1 = np.conj(vector[self.dual_matrix_rows[idx_cl]])
                        v2 = vector[self.dual_matrix_cols[idx_cl]]
                        coefs = np.real((self.MO_transpose[idx_cl]).dot(v1*v2))
    
                        eigenvectors[self.eigencuts_counter] = vector
                        eigencuts_coefs[self.eigencuts_counter] = coefs
                        eigencuts_idx[self.eigencuts_counter] =(idx_cl)
                        eigencuts_dual[self.eigencuts_counter] = s[i]
                        eigencuts_slack[self.eigencuts_counter] = 0
                        eigencuts_sstep[self.eigencuts_counter] = self.serious_step_number
                        self.eigencuts_counter+=1
           
        self.eigenvectors, self.eigencuts_coefs, self.eigencuts_idx, self.eigencuts_dual, self.eigencuts_slack,self.eigencuts_sstep = eigenvectors, eigencuts_coefs, eigencuts_idx, eigencuts_dual, eigencuts_slack, eigencuts_sstep
        
    """QP solution """
    def __solveQP(self,maxiter):
        betacuts_keys, eigencuts_keys = list(self.betacuts_coefs), list(self.eigencuts_coefs)
        betacuts_offset, betacuts_idx,betacuts_dual = [self.betacuts_offset[key] for key in betacuts_keys],[self.betacuts_idx[key] for key in betacuts_keys],np.array([self.betacuts_dual[key] for key in betacuts_keys])
        eigencuts_idx, eigencuts_dual = [self.eigencuts_idx[key] for key in eigencuts_keys], np.array([self.eigencuts_dual[key] for key in eigencuts_keys])
        kfixed_cuts, kbetacuts, keig_cuts = len(self.fixed_cuts), len(betacuts_keys), len(eigencuts_keys)
        tconcat = t0 = time.time()
        coefs = np.concatenate([self.eigencuts_coefs[key] for key in eigencuts_keys])
        y = np.concatenate([self.vars[idx_clique] for idx_clique in eigencuts_idx])
        x = np.concatenate([np.ones(len(self.vars[idx_clique])) * k for k,idx_clique in enumerate(eigencuts_idx)])
        Meigencuts = coo_matrix((coefs,(x,y)),shape= (keig_cuts,self.d)).tocsc()
        coefs = np.array([self.betacuts_coefs[key] for key in betacuts_keys])
        y = self.N+np.array(betacuts_idx)
        x = np.arange(kbetacuts)
        Mbetacuts = coo_matrix((coefs,(x,y)),shape= (kbetacuts,self.d)).tocsc()
        M = vstack([self.Mfixedcuts, Mbetacuts, Meigencuts])
        q = M.dot(self.thetabar)
        q[kfixed_cuts:kfixed_cuts+kbetacuts] = q[kfixed_cuts:kfixed_cuts+kbetacuts] + np.array(betacuts_offset)
        self.concattime =  tconcat - time.time()
        
        t0 = time.time()
        Mprime = self.invHessian.dot(M.T)
        gram = M.dot(Mprime)
        self.gramtime = time.time()-t0
        totcutnumber = kbetacuts +keig_cuts + kfixed_cuts
        t0 = time.time()
        A0 = identity(totcutnumber)
        A_betacuts = coo_matrix(([1]*kbetacuts,(betacuts_idx,[kfixed_cuts+i for i in range(kbetacuts)])),shape = (self.n,totcutnumber)).tocsc()
        A_eigen_cuts = coo_matrix(([1]*keig_cuts,(eigencuts_idx,[kfixed_cuts+kbetacuts+i for i in range(keig_cuts)])),shape = (self.cliques_nbr,totcutnumber)).tocsc()
        A = vstack([A0, hstack([self.A_fixed_cuts, csc_matrix((self.A_fixed_cuts.shape[0],totcutnumber - kfixed_cuts))]), A_betacuts, A_eigen_cuts])
        l = np.array(([0]*totcutnumber) + ([1]*(self.N+2*self.n+2*self.cl)) + ([0]*self.cliques_nbr)) 
        u = np.concatenate([np.ones(totcutnumber)*np.inf,np.ones(self.N+2*self.n+2*self.cl),np.array([self.rho[idx_clique] for idx_clique in range(self.cliques_nbr)])]) 
        
        m = osqp.OSQP()
        m.setup(P= gram.tocsc() , q=q, A=A.tocsc(), l=l, u=u,eps_rel = osqp_eps_rel,polish = osqp_polish,verbose=osqp_verbose, eps_dual_inf  = osqp_eps_dual, max_iter = maxiter,check_termination =100)#,linsys_solver = "mkl pardiso")
        
        if self.it>=1:
            theta_0 = np.concatenate([self.fixed_cuts_dual, betacuts_dual, eigencuts_dual])
            slacks = np.concatenate([self.fixed_cuts_slack,np.array([self.betacuts_slack[key] for key in betacuts_keys]),np.array([self.eigencuts_slack[key] for key in eigencuts_keys])])
            y0 = np.concatenate([slacks,self.tfixedcuts,self.tbeta,0.5*(self.teigenvalue+self.errors_by_clique)])
            m.warm_start(x=theta_0,y = y0)
            
        results= m.solve()
        vector = results.x
        vectorbis = vector.copy()
        UB = self.offset+ 0.5*(gram.dot(vectorbis).dot(vectorbis)) + q.dot(vectorbis)
        
        counter=0
        while UB<self.current_value_bar and counter<4 :#or infeas>1E-3:
            m.update_settings(max_iter=min(2000,maxiter))
            results= m.solve()
            vector = results.x
            vectorbis = vector.copy()
            UB = self.offset+ 0.5*(gram.dot(vectorbis).dot(vectorbis)) + q.dot(vectorbis)
            counter+=1
        
        if (UB<self.current_value_bar):
            print('OSQP encounters numerical difficulties')
            UB = self.current_value_bar+10
            self.kappa = self.kappa*(increase_ratio**2)
            self.hessian = self.kappa*identity(self.d)
            self.invHessian = (1/self.kappa)*identity(self.d)
            
        self.qptime =  time.time()-t0
        self.fixed_cuts_dual=  vector[:kfixed_cuts]
        self.betacuts_dual = {key: vector[kfixed_cuts+aux] for aux,key in enumerate(betacuts_keys)}
        self.eigencuts_dual = {key: vector[kfixed_cuts+kbetacuts+aux] for aux,key in enumerate(eigencuts_keys)}
               
        self.fixed_cuts_slack = results.y[:kfixed_cuts]
        self.betacuts_slack = {key : results.y[kfixed_cuts+aux] for aux, key in enumerate(betacuts_keys)}
        self.eigencuts_slack = {key : results.y[kfixed_cuts+kbetacuts+aux] for aux, key in enumerate(eigencuts_keys)}
        
        
        theta = self.thetabar + (Mprime).dot(vector)
        value_betacuts = Mbetacuts.dot(theta) + np.array(betacuts_offset)
        self.tbeta = np.ones(self.n)*np.inf
        for aux in range(len(value_betacuts)):
            i = betacuts_idx[aux]
            self.tbeta[i] = min(value_betacuts[aux],self.tbeta[i])
        value_eigen_cuts = Meigencuts.dot(theta)
        self.teigenvalue = np.zeros(self.cliques_nbr)
        for aux in range(len(value_eigen_cuts)):
            i = eigencuts_idx[aux]
            self.teigenvalue[i] = min(value_eigen_cuts[aux],self.teigenvalue[i])
        value_fixed_cuts = self.Mfixedcuts.dot(theta)
        self.tfixedcuts = np.array([min(value_fixed_cuts[2*i],value_fixed_cuts[2*i+1]) for i in range(len(value_fixed_cuts)//2)])
        
        obj_value_theta = self.offset-0.5*(theta-self.thetabar).dot(self.hessian.dot(theta-self.thetabar))+self.tbeta.sum()+self.tfixedcuts.sum()+np.array([self.rho[idx_clique]*self.teigenvalue[idx_clique] for idx_clique in range(self.cliques_nbr)]).sum()
        self.grad_norm = self.kappa * np.linalg.norm(theta-self.thetabar)
        self.thetaval = theta
        self.alpha_val = theta[:self.N]
        self.beta_val = theta[self.N:self.N+self.n]
        self.gamma_val = theta[self.N+self.n:self.N+2*self.n]
        self.lambda_f_val = theta[self.N+2*self.n :self.N+2*self.n+self.cl ]
        self.lambda_t_val = theta[self.N+2*self.n+self.cl : self.N+2*self.n+2*self.cl ]
        self.eta_val = theta[self.N+2*self.n+2*self.cl:]
        return obj_value_theta,max(obj_value_theta,UB)
    
    
    def __initialize_G_cutting_planes(self):
        #Constraints beta
        self.fixed_cuts = []
        self.betacuts_coefs, self.betacuts_idx, self.betacuts_offset,self.betacuts_dual,self.betacuts_slack = {},{},{},{},{}
        self.betacuts_counter = 0
        
        offset = 0
        
        for idx_clique in range(self.cliques_nbr):
            clique = self.cliques[idx_clique]
            for global_indx in clique:
                local_index = self.localBusIdx[idx_clique,global_indx]
                self.fixed_cuts.append(coo_matrix(([-(self.Vmin[global_indx]**2)],([0],[offset+local_index])),shape = (1,self.d)).tocsc())
                self.fixed_cuts.append(coo_matrix(([-(self.Vmax[global_indx]**2)],([0],[offset+local_index])),shape = (1,self.d)).tocsc())
            offset+=len(clique)
        
        del offset 
        
        #Constraints gamma
        for i in range(self.n):
            self.fixed_cuts.append(coo_matrix(([self.Qload[i] - self.sumQmin[self.buslist[i]]],([0],[self.N+self.n+i])),shape = (1,self.d)).tocsc())
            self.fixed_cuts.append(coo_matrix(([self.Qload[i] - self.sumQmax[self.buslist[i]]],([0],[self.N+self.n+i])),shape = (1,self.d)).tocsc())
            
        #Constraints lambda
        for idx_line in range(self.cl):
            self.fixed_cuts.append(coo_matrix(([0],([0],[self.N+2*self.n+idx_line])),shape = (1,self.d)).tocsc())
            self.fixed_cuts.append(coo_matrix(([-(self.Imax[idx_line]**2)/self.scaling_lambda_f[idx_line]],([0],[self.N+2*self.n+idx_line])),shape = (1,self.d)).tocsc())
            self.fixed_cuts.append(coo_matrix(([0],([0],[self.N+2*self.n+self.cl+idx_line])),shape = (1,self.d)).tocsc())
            self.fixed_cuts.append(coo_matrix(([-(self.Imax[idx_line]**2)/self.scaling_lambda_t[idx_line]],([0],[self.N+2*self.n+self.cl+idx_line])),shape = (1,self.d)).tocsc())
        k = 2*(self.N + self.n + 2*self.cl) 
        self.A_fixed_cuts = coo_matrix(([1] * k, ([i//2 for i in range(k)],[i for i in range(k)])), shape = (k//2, k)).tocsc()
        self.Mfixedcuts = vstack(self.fixed_cuts)
        
        
        grad1 = self.__G_gradient_beta(self.beta_val)
        value_function_beta = self.__G_value_beta(self.beta_val)
        for i in range(self.n):
            self.__add_betacut(grad1[i],value_function_beta[i],i)  
    
                                               
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
    
    def set_inital_values(self,alpha, beta, gamma, lambda_f, lambda_t, eta):
        """Setting initial variables for a warm-start """
        self.alpha_val = self.alpha_ref =  alpha
        self.beta_val = self.beta_ref = beta
        self.gamma_val = self.gamma_ref = gamma 
        self.lambda_f_val = self.lambda_f_ref = lambda_f
        self.lambda_t_val = self.lambda_t_ref = lambda_t
        self.eta_val = self.eta_bar =  eta
        self.thetabar = self.thetaval = np.concatenate([self.alpha_val, self.beta_val, self.gamma_val,self.lambda_f_val,self.lambda_t_val, self.eta_val])
        self.initial_values_set = True
        
    def solve(self,kappa0):
        """
        Main function
        Method to solve the dual relaxation with a proximal bundle method 
        """
        #### Initialize parameters and auxiliary variables
        self.it,self.serious_step_number = 0,0
        self.eigencuts_sstep = {key : 0 for key in self.eigencuts_coefs}
        self.cut_added = 0
        self.consecutive_null_step = 0
        self.delta = np.inf
        OSQPmaxiter = 500 
        mbundle, rel_tol,maxit = self.config["mbundle"],self.config["rel_tol"],self.config["maxit"]
        self.mbundle, self.maxit = mbundle, maxit
        ratio_added_cuts = self.config['ratio_added_cuts']
        self.kappa = kappa0
        self.mu, self.last_grad = {},{}
        for idx_cl in range(self.cliques_nbr):
            self.mu[idx_cl] = kappa0
        ub_for_infeas_detection = self.__compute_upper_bound()
        values_logger = []
        #### Initialize stability center
        self.warmstart = True
        if not(self.initial_values_set):
            self.alpha_val = self.alpha_ref =  magnitude_init_perturb*(2*np.random.rand(self.N)-1)
            val_init_beta = (1+magnitude_init_perturb*np.random.rand(self.n))*np.mean(self.lincost) #np.min(self.lincost) #+ np.std(self.lincost)
            self.beta_val = self.beta_ref = val_init_beta 
            val_init_gamma = magnitude_init_perturb*(2*np.random.rand(self.n)-1)
            self.gamma_val = self.gamma_ref = val_init_gamma 
            self.lambda_f_val = self.lambda_f_ref = np.zeros(self.cl)
            self.lambda_t_val = self.lambda_t_ref = np.zeros(self.cl)
            self.eta_val = self.eta_bar =  np.zeros(self.eta_nbr)
            self.thetabar = self.thetaval = np.concatenate([self.alpha_val, self.beta_val, self.gamma_val,self.lambda_f_val,self.lambda_t_val, self.eta_val])
            self.initial_values_set, self.warmstart = True, False
        
        ##############################################################################################
                 
        self.__initialize_G_cutting_planes()
        
        Fval = 0
        for idx_clique in range(self.cliques_nbr):
            U,s = self.__SVD(self.thetaval[self.vars[idx_clique]],idx_clique)
            Fval+= self.rho[idx_clique]* min(0,s.min())
            self.__add_eigencuts(U,s,idx_clique)
            
        self.error_bar = self.error = Fval
        self.Gval = self.__G_value_oracle(self.alpha_val, self.beta_val, self.gamma_val, self.lambda_f_val, self.lambda_t_val)
        self.current_value_bar = self.current_value = self.Gval + Fval
        del Fval
        self.best_value= self.current_value
        self.best_certified_value = self.certified_value(self.alpha_val, self.beta_val, self.gamma_val, self.lambda_f_val, self.lambda_t_val, self.eta_val)
        self.Gval_bar = self.Gval
        self.hessian = self.kappa*identity(self.d)
        self.invHessian = (1/self.kappa)*identity(self.d)

        
        self.finished = False
        estimation = self.__estimation()
        self.tol =tol= estimation*rel_tol
        self.__initial_info()
        self.bmtime = 0
        
        while self.it<maxit:
            #print("--------------Iteration number {0}------------------".format(self.it))
            #print("Max iter = {0}".format(OSQPmaxiter))
            LB,UB = self.__solveQP(OSQPmaxiter)
            if (UB-LB)>0.95*(UB-self.current_value_bar):
                OSQPmaxiter=2500
            elif (UB-LB)>0.7*(UB-self.current_value_bar):
                OSQPmaxiter=1000
            else:
                OSQPmaxiter=500
            #Computation of delta and stopping test
            self.delta = UB - self.current_value_bar
            self.deltaSStep = max(tol,0.5*(LB+UB) - self.current_value_bar)
            
            if self.delta<10*tol:
                ratio_added_cuts = ratio_added_cuts_end
                OSQPmaxiter=max(OSQPmaxiter,2000)
            t0 = time.time()
            self.Gval = self.__G_value_oracle(self.alpha_val, self.beta_val, self.gamma_val, self.lambda_f_val, self.lambda_t_val)
            #Computation of the function's value
            t0 = time.time()
            U,s,Fval = {},{},0
            self.errors_by_clique = [0]*self.cliques_nbr
            for idx_clique in range(self.cliques_nbr):
                U[idx_clique],s[idx_clique] = self.__SVD(self.thetaval[self.vars[idx_clique]],idx_clique)
                Fval+= self.rho[idx_clique]* min(0,s[idx_clique].min())
                self.errors_by_clique[idx_clique] = min(0,s[idx_clique].min())
            self.oracleTime = time.time()-t0
            self.error = Fval
            self.current_value = self.Gval+ Fval
            self.best_value = max(self.current_value,self.best_value)
            self.best_certified_value = max(self.certified_value(self.alpha_val, self.beta_val, self.gamma_val, self.lambda_f_val, self.lambda_t_val, self.eta_val),self.best_certified_value)
            values_logger.append(self.best_certified_value)
            if self.current_value>ub_for_infeas_detection:
                self.__info(True)
                if self.verbose:
                    print("-----------------------------------------------------------------")
                    print("Infeasible primal problem")
                    print("-----------------------------------------------------------------")
                self.__log()
                self.__final_status_log('Converged')
                return 
                
           
            t0 = time.time()
            if self.delta<tol:
                self.finished = True
                self.__log()
                self.__info(True)
                if self.verbose:
                    print("-----------------------------------------------------------------")
                    print("Reached termination criteria after {0} iterations. \nBest value found is {1}".format(self.it,"%.7g" % self.best_certified_value))
                    print("-----------------------------------------------------------------")
                self.__log()
                self.__final_status_log('Converged')
                return self.best_value
            #Serious-step or null-step
            if (self.current_value - self.current_value_bar >= self.mbundle*self.deltaSStep):
                if self.serious_step_number%3==2:
                     self.__delete_betacuts()
                     self.__delete_eigencuts()
                    
                if self.delta>10*tol and self.config["aggregation"] and self.serious_step_number%8==7:
                      self.__aggregate_eigencuts()
                      self.kappa = self.kappa*(increase_ratio**2)
                                     
                if self.consecutive_null_step<=2:
                    self.kappa = self.kappa*decrease_ratio
                    self.kappa = max(self.kappa,1E-7)
                self.step_type = "Serious step"
                
                self.hessian = self.kappa*identity(self.d)
                self.invHessian = (1/self.kappa)*identity(self.d)
                
                #Update stability center
                self.error_bar = self.error
                self.Gval_bar = self.Gval
                self.current_value_bar = self.current_value
                self.alpha_ref = self.alpha_val 
                self.beta_ref = self.beta_val
                self.gamma_ref = self.gamma_val
                self.lambda_f_ref = self.lambda_f_val
                self.lambda_t_ref = self.lambda_t_val
                self.eta_ref = self.eta_val
                self.thetabar = self.thetaval
                self.serious_step_number+=1
                self.consecutive_null_step = 0
                 
            else:
                
                if self.consecutive_null_step>=nb_null_step_before_increase:
                    if self.consecutive_null_step%nb_null_step_before_increase==0:
                        self.kappa = self.kappa*increase_ratio
                        self.hessian = self.kappa*identity(self.d)
                        self.invHessian = (1/self.kappa)*identity(self.d)
                                              
                if (self.consecutive_null_step>=max_number_of_consecutive_nsteps and self.delta<100*tol) or self.__stall_condition(values_logger):
                    self.finished = True
                    self.__info(True)
                    if self.verbose:
                        print("-----------------------------------------------------------------")
                        print("Solver stopped (stall).\nBest value found is {0}".format("%.7g" % self.best_certified_value))
                        print("-----------------------------------------------------------------")
                    self.__log()
                    self.__final_status_log('Stall')
                    return
                self.step_type = "Null step"
                self.consecutive_null_step+= 1
                            
            #Add cutting planes
            grad_beta = self.__G_gradient_beta(self.beta_val)
            value_function_beta = self.__G_value_beta(self.beta_val)
            liste1 = [(self.rho[idx_clique]*self.teigenvalue[idx_clique]-self.rho[idx_clique]* min(0,s[idx_clique].min())) for idx_clique in range(self.cliques_nbr)]
            liste2 = [(self.tbeta[i]-value_function_beta[i]) for i in range(self.n)]
            liste = liste1+liste2
            somme = sum(liste)
            tuples1 = [(liste1[idx_clique],idx_clique,'cl') for idx_clique in range(self.cliques_nbr)]
            tuples2 = [(liste2[i],i,'beta') for i in range(self.n)]
            tuples = tuples1 + tuples2
            tuples.sort(key=(operator.itemgetter(0)), reverse = True)
            idx = argmin_cumsum(tuples,ratio_added_cuts*somme)
            nbr_beta_cut = 0
            for aux in range(idx+1):
                if tuples[aux][2]=='cl':
                    idx_clique = tuples[aux][1]
                    self.__add_eigencuts(U[idx_clique],s[idx_clique],idx_clique) 
                else:
                    nbr_beta_cut+=1
                    idx = tuples[aux][1]
                    self.__add_betacut(grad_beta[idx],value_function_beta[idx],idx)
            
            self.bmtime = time.time() - t0
            self.__info()
            self.__log()
            self.it+=1
        self.finished = True
        self.__info(True)
        if self.verbose:
            print("-----------------------------------------------------------------")
            print("Max number of iterations.\nBest value found is {0}".format("%.7g" % self.best_certified_value))
            print("-----------------------------------------------------------------")
        self.__log() 
        self.__final_status_log('Max. number of iterations.')
        