#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 14:41:27 2023

@author: watsonlevens
"""
import pickle 
import numpy as np


def degrees_and_theoreticalP_k(n_q ,q, max_k):  ## Degrees and PDF for the theoretical calculations
    '''
    max_k : maximum degree value for n realization of the fof function)
    n_q : Number of neighbours
    q : Probability of a new node to attach to neighbouring nodes
    '''
    k=np.arange(2 + n_q,max_k) 
    a = 2*(1+(q*n_q))
    b = a/(q*n_q)
    
    l =b*(a+b)**b 
    P_k=l*(k+b)**(-(1+b))
    
    
    ## Combined Eqn, equation as whole
    #P_k=((2*(1+(q*n_q))/(q*n_q))*((2*(1+(q*n_q)))+(2*(1+(q*n_q))/(q*n_q)))**(2*(1+(q*n_q))/(q*n_q)))*((k+(2*(1+(q*n_q))/(q*n_q)))**(-(1+(2*(1+(q*n_q)))/(q*n_q))))
    return k, P_k 


n_q=2
q = 1
max_k = 1000 ## maximum degree
kt_and_Pkt = degrees_and_theoreticalP_k(n_q ,q, max_k)
k_theoretical= kt_and_Pkt[0]
Pk_theoretical=kt_and_Pkt[1]

### Saving   
fileName   =  'TheoreticalDegree_and_Pk' + ".pkl";
file_pi = open(fileName, 'wb') 
pickle.dump(kt_and_Pkt, file_pi)
