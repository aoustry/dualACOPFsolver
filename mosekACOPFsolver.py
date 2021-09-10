# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 11:28:55 2021

@author: aoust
"""


import itertools
from scipy.sparse import lil_matrix, csc_matrix, coo_matrix, diags, save_npz, load_npz, identity
import numpy as np
from mosek.fusion import *

myZeroforCosts = 1E-6
scale = 0.0001

class MosekRelaxationSolver():
    
    def __init__(self, ACOPF):
        self.name = ACOPF.name
        self.n, self.gn, self.m, self.cl = ACOPF.n, ACOPF.gn, ACOPF.gn, ACOPF.cl
        self.Vmin, self.Vmax = ACOPF.Vmin, ACOPF.Vmax
        self.Pmin,self.Pmax,self.Qmin, self.Qmax = ACOPF.Pmin,ACOPF.Pmax,ACOPF.Qmin, ACOPF.Qmax
        self.offset, self.lincost, self.quadcost = ACOPF.offset, np.array(ACOPF.lincost), ACOPF.quadcost
        self.buslist, self.buslistinv,self.genlist = ACOPF.buslist, ACOPF.buslistinv, ACOPF.genlist
        self.cliques, self.ncliques, self.cliques_nbr = ACOPF.cliques, ACOPF.ncliques, ACOPF.cliques_nbr
        self.cliques_parent, self.cliques_intersection, self.localBusIdx = ACOPF.cliques_parent, ACOPF.cliques_intersection, ACOPF.localBusIdx
        self.Pload, self.Qload = np.array(ACOPF.Pload), np.array(ACOPF.Qload) 
        self.M = ACOPF.M
        self.HM, self.ZM = ACOPF.HM, ACOPF.ZM
        self.Nt, self.Nf = ACOPF.Nt, ACOPF.Nf
        self.Imax = ACOPF.Imax
        
        self.bus_to_gen = {}
        for idx in range(self.n):
            self.bus_to_gen[idx] = []
        for idx_gen,gen in enumerate(self.genlist):
            bus,index = self.genlist[idx_gen]
            index_bus =  self.buslistinv[bus]
            self.bus_to_gen[index_bus].append(idx_gen)
            
        self.cliques_contribution = {}
        for index_bus in range(self.n):
            self.cliques_contribution[index_bus] = set()
       
        for couple in self.M:
            clique,index_bus = couple
            self.cliques_contribution[index_bus].add(clique)
    
    
    def Joperator(self,matrix):
        A = np.real(matrix)
        B = np.imag(matrix)
        line1 = np.hstack([A, -B])
        line2 = np.hstack([B, A])
        return (1/(np.sqrt(2)))*np.vstack([line1,line2])

                               
    
    def solve(self):
        """
        Method to solve the real formulation of the rank relaxation

        Returns
        -------
        None.

        """
        
        with Model("OPF-rank-relaxation") as M:
            #Upper level var
            Pgen = M.variable("Pgen", self.gn, Domain.unbounded())
            aux = M.variable("aux", self.gn+2, Domain.inRotatedQCone())
            Qgen = M.variable("Qgen", self.gn, Domain.unbounded())
            
            #Objective
            M.objective( ObjectiveSense.Minimize, Expr.add(Expr.add(scale*self.offset,Expr.dot(scale*self.lincost,Pgen)),Expr.mul(2,aux.index(1))))
            M.constraint(aux.index(0), Domain.equalsTo(1))
            M.constraint(Expr.sub(aux.pick(range(2,self.gn+2)),Expr.mulElm([np.sqrt(scale*cost) for cost in self.quadcost],Pgen)), Domain.equalsTo(0,self.gn))
            
            ###Active Power bounds
            M.constraint(Pgen,Domain.greaterThan(np.array([self.Pmin[idx] for idx in range(self.gn)])))
            M.constraint(Pgen,Domain.lessThan(np.array([self.Pmax[idx] for idx in range(self.gn)])))
            
            ###Reactive Power bounds
            M.constraint(Qgen,Domain.greaterThan(np.array([self.Qmin[idx] for idx in range(self.gn)])))
            M.constraint(Qgen,Domain.lessThan(np.array([self.Qmax[idx] for idx in range(self.gn)])))
            
            X,A,B = {},{},{}
            constr_mag_low,constr_mag_up = {},{}
            for idx_clique in range(self.cliques_nbr):
                nc = self.ncliques[idx_clique]
                clique = self.cliques[idx_clique]
                X[idx_clique] = M.variable(Domain.inPSDCone(2*nc))
                A[idx_clique] = M.variable("A"+str(idx_clique), [nc,nc], Domain.unbounded())
                B[idx_clique] = M.variable("B"+str(idx_clique), [nc,nc], Domain.unbounded())
                # # #Voltage bounds
                constr_mag_low[idx_clique] = M.constraint(A[idx_clique].diag(),Domain.greaterThan(np.array([self.Vmin[idx]**2 for idx in clique])))
                constr_mag_up[idx_clique] = M.constraint(A[idx_clique].diag(),Domain.lessThan(np.array([self.Vmax[idx]**2 for idx in clique])))
            
                #Link between isometry matrix X and matrices A and B
                M.constraint(Expr.sub(Expr.mul(1/np.sqrt(2),A[idx_clique]),X[idx_clique].slice([0,0], [nc,nc])), Domain.equalsTo(0,nc,nc))
                M.constraint(Expr.sub(Expr.mul(1/np.sqrt(2),A[idx_clique]),X[idx_clique].slice([nc,nc], [2*nc,2*nc])), Domain.equalsTo(0,nc,nc))
                M.constraint(Expr.sub(Expr.mul(1/np.sqrt(2),B[idx_clique]),X[idx_clique].slice([nc,0], [2*nc,nc])), Domain.equalsTo(0,nc,nc))
                M.constraint(Expr.add(Expr.mul(1/np.sqrt(2),B[idx_clique]),X[idx_clique].slice([0,nc], [nc,2*nc])), Domain.equalsTo(0,nc,nc))
                        
            #Active and Reactive Power conservation
            active_pwr_constr, reactive_pwr_constr = {},{}
            for index_bus in range(self.n):
                
                sumPgen = Expr.zeros(1)
                sumQgen = Expr.zeros(1)
                for i in self.bus_to_gen[index_bus]:
                    sumPgen = Expr.add(sumPgen, Pgen.index(i))
                    sumQgen = Expr.add(sumQgen, Qgen.index(i))
                
                #JHMbus, JiZMbus = {},{}
                Ptransfer, Qtransfer = Expr.zeros(1),Expr.zeros(1)
                
                for idx_clique in self.cliques_contribution[index_bus]:
                    nc = self.ncliques[idx_clique]
                    auxHM = self.Joperator(self.HM[idx_clique,index_bus].toarray())
                    auxiZHM = self.Joperator(1j*(self.ZM[idx_clique,index_bus]).toarray())
                    auxHM = coo_matrix(auxHM)
                    auxHM.eliminate_zeros()
                    auxiZHM = coo_matrix(auxiZHM)
                    auxiZHM.eliminate_zeros()
                    JHMbus = Matrix.sparse(2*nc,2*nc,auxHM.row, auxHM.col, auxHM.data)
                    JiZMbus = Matrix.sparse(2*nc,2*nc,auxiZHM.row, auxiZHM.col, auxiZHM.data)
                    Ptransfer = Expr.add(Ptransfer,Expr.dot(JHMbus,X[idx_clique]))   
                    Qtransfer = Expr.add(Qtransfer,Expr.dot(JiZMbus,X[idx_clique]))  
                if len(self.bus_to_gen[index_bus])>0:
                    active_pwr_constr[index_bus] = M.constraint(Expr.sub(sumPgen,Ptransfer),Domain.equalsTo(self.Pload[index_bus]))
                    reactive_pwr_constr[index_bus] = M.constraint(Expr.sub(sumQgen,Qtransfer),Domain.equalsTo(self.Qload[index_bus]))
                else:
                    active_pwr_constr[index_bus] = M.constraint(Expr.neg(Ptransfer),Domain.equalsTo(self.Pload[index_bus]))
                    reactive_pwr_constr[index_bus] = M.constraint(Expr.neg(Qtransfer),Domain.equalsTo(self.Qload[index_bus]))                   
            
            #Lines intensity constraints
            nf_constraint,nt_constraint = {},{}
            for idx_clique, idx_line in self.Nt:
                nc = self.ncliques[idx_clique]
                Nf = self.Joperator(self.Nf[idx_clique,idx_line].toarray())
                Nf = coo_matrix(Nf)
                Nf.eliminate_zeros()
                Nf = Matrix.sparse(2*nc,2*nc,Nf.row, Nf.col, Nf.data)
                Nt = self.Joperator(self.Nt[idx_clique,idx_line].toarray())
                Nt = coo_matrix(Nt)
                Nt.eliminate_zeros()
                Nt = Matrix.sparse(2*nc,2*nc,Nt.row, Nt.col, Nt.data)
                nf_constraint[idx_line] = M.constraint(Expr.dot(Nf,X[idx_clique]), Domain.lessThan(self.Imax[idx_line]**2))
                nt_constraint[idx_line] = M.constraint(Expr.dot(Nt,X[idx_clique]), Domain.lessThan(self.Imax[idx_line]**2))
            
            #Overlapping constraints
            overlapping_constr = {}
            aux = 0
            for clique_idx in range(self.cliques_nbr):
                nc = self.ncliques[clique_idx]
                clique_father_idx = self.cliques_parent[clique_idx]
                for global_idx_bus_b in self.cliques_intersection[clique_idx]:
                    local_index_bus_b = self.localBusIdx[clique_idx,global_idx_bus_b]
                    local_index_bus_b_father = self.localBusIdx[clique_father_idx,global_idx_bus_b]
                    overlapping_constr[aux] = M.constraint(Expr.sub(A[clique_idx].index(local_index_bus_b,local_index_bus_b),A[clique_father_idx].index(local_index_bus_b_father,local_index_bus_b_father)), Domain.equalsTo(0.0))
                    aux+=1
                for global_idx_bus_b,global_idx_bus_a in itertools.combinations(self.cliques_intersection[clique_idx], 2):
                    local_index_bus_b,local_index_bus_a = self.localBusIdx[clique_idx,global_idx_bus_b],self.localBusIdx[clique_idx,global_idx_bus_a]
                    local_index_bus_b_father,local_index_bus_a_father = self.localBusIdx[clique_father_idx,global_idx_bus_b],self.localBusIdx[clique_father_idx,global_idx_bus_a]
                    overlapping_constr[aux] =M.constraint(Expr.sub(A[clique_idx].index(local_index_bus_b,local_index_bus_a),A[clique_father_idx].index(local_index_bus_b_father,local_index_bus_a_father)), Domain.equalsTo(0.0))
                    aux+=1
                    overlapping_constr[aux] =M.constraint(Expr.sub(B[clique_idx].index(local_index_bus_b,local_index_bus_a),B[clique_father_idx].index(local_index_bus_b_father,local_index_bus_a_father)), Domain.equalsTo(0.0))
                    aux+=1
                    
            M.setLogHandler(open("output/"+self.name+"_mosek.txt","w"))            # Add logging
            M.acceptedSolutionStatus(AccSolutionStatus.Anything)
            M.setSolverParam("intpntCoTolPfeas", 1.0e-10)
            M.setSolverParam("intpntCoTolDfeas", 1.0e-10)
            M.solve()
            
            alpha = []
            for idx_clique in range(self.cliques_nbr):
                vec_up, vec_low = constr_mag_up[idx_clique].dual(),constr_mag_low[idx_clique].dual()
                alpha = alpha+ [vec_low[i]-vec_up[i] for i in range(self.ncliques[idx_clique])]
            beta = [active_pwr_constr[index_bus].dual() for index_bus in range(self.n)]
            gamma = [reactive_pwr_constr[index_bus].dual() for index_bus in range(self.n)]
            lamda_f = [-nf_constraint[idx_line].dual() for idx_line in range(self.cl)]
            lamda_t = [-nt_constraint[idx_line].dual() for idx_line in range(self.cl)]
            nu = [-overlapping_constr[aux].dual() for aux in overlapping_constr]
            
            return M.dualObjValue()*1/scale,np.array(alpha)/scale, np.array(beta).reshape(self.n)/scale, np.array(gamma).reshape(self.n)/scale, np.array(lamda_f).reshape(self.cl)/scale, np.array(lamda_t).reshape(self.cl)/scale, np.array(nu).reshape(len(nu))/scale