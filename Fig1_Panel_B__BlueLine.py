#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 11:17:13 2021
@author: watsonlevens
"""
import pickle 
import networkx as nx
import random 

def FOF_L(m0,q,N):
    '''
    N : Network size
    q : Probability of attaching to neighbour nodes (q must be <1 for ploting degree distribution)
    m0 : Initial number of nodes.
    '''
    G=nx.empty_graph(create_using=nx.Graph())  
    G.add_node(0)                   
    for source in range(m0, N):                                                                 
        nodes = [nod for nod in G.nodes()] 
        node = random.choice(nodes) 
        neighbours = [nbr for nbr in G.neighbors(node) 
                          if not G.has_edge(source, nbr) 
                          and not nbr == source]  
        G.add_node(source) 
        G.add_edge(source, node) 
        while len(neighbours)>0: 
           nbr = neighbours.pop() 
           if q >random.random():  
              G.add_edge(source, nbr)
    return G 

def get_all_degree(G):  
    all_in_degrees= G.degree() 
    all_deg= list(nod_degres for nod, nod_degres in all_in_degrees)
    return all_deg


 
N = 1000 
q = 0.8 
m0 = 1 
all_all_deg=[] 
n=10 # Number of realization 
for rep in range(n): 
    #print('Realisation number: ',rep)
    Network=FOF_L(m0,q,N)
    all_deg=get_all_degree(Network)
    all_all_deg.extend(all_deg) 

## Saving the degrees        
######################################################################################################################
TheorfileName   =  'FOF_L_all_all_degq8' + ".pkl";
Theofile_pi = open(TheorfileName, 'wb') 
pickle.dump(all_all_deg, Theofile_pi)
######################################################################################################################



