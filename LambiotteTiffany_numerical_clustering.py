#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 11:17:13 2021

@author: watsonlevens
"""
import pickle 
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
from networkx.utils import not_implemented_for
from networkx.utils import py_random_state

# A function to generate Lambiotte model  
def Lambiotte_model(m0,q,N):
    G=nx.empty_graph(create_using=nx.Graph())  # Initial empty graph         
    G.add_node(0)  # Add node 0 to the empty network as the initial node.  
                 
    for source in range(m0, N): # Start network formation from m0 node and stop at N node
        #################################################################################################################                    
        # Step1. Pick a node randomly and make connection.
        # ########################################################################################################################                                                                   
        nodes = [nod for nod in G.nodes()] # Existing nodes in the network
        node = random.choice(nodes) # Choose a random target node
        neighbours = [nbr for nbr in G.neighbors(node) # neighbors of the node
                          if not G.has_edge(source, nbr) # If no edge btn source node and nbr node
                          and not nbr == source] # If neighbor node is not equal to source node 
        G.add_node(source) ## Add node to the network # SOURCE MUST BE ADDED AFTER RANDOM SAMPLING SO AS 
                           ## TO AVOID SELF LOOP WHEN ADDING EDGES
        G.add_edge(source, node) # Add edge 
        
        #################################################################################################################                    
        # Step2. With prob q, make connection with all nbrs.
        #########################################################################################################################                                                                             
        while len(neighbours)>0: # all neighbours need to make connection with the new node
           nbr = neighbours.pop() #
           if q >random.random():  
              G.add_edge(source, nbr) # Add edge 
    return G 

################################################################################
@py_random_state(2)
@not_implemented_for('directed') 
def node_clustering(G, trials=1000, seed=None):
    #Calculates  [trials] weakly connected triangle clusters in graph G
    n = len(G)
    triangles = 0
    nodes = list(G)
    
    #List of all degree, triangle pairs
    degree_and_triangles=[]
    
    
    for i in [int(seed.random() * n) for i in range(trials)]:
        #print('iteration no:',i)
        #Can take same node twice or more. Can be a problem on small graphs.
        #FOLLOWERS AND FOLLOWED, Weakly connected
        nbrs = list(G[nodes[i]])
        #print('Neighbors of ',i,'are',list(nbrs))
        degree=len(nbrs)
        
        triangles=0
        
#            
        for j,u in enumerate(nbrs):
            #Find all pairs
            for v in nbrs[j+1:]:
                if (u in G[v]):
                    #Weakly connected
                    triangles += 1
        if degree > 1:
            degree_and_triangles.append([degree,(triangles/(degree*(degree-1)/2))])  
        else:
            degree_and_triangles.append([degree,0])
            
    return degree_and_triangles 

N = 1000# Network size # population size
q = 0.5 # Probability of attaching to neighbour nodes (q must be <1 for ploting degree distribution)
m0 = 1 # Initial node.

Network= Lambiotte_model(m0,q,N)
Degrees_and_clustering_coefficients =node_clustering(Network,trials=1000)
array_Degrees_and_clustering_coefficients=np.array(Degrees_and_clustering_coefficients)
clustering_coefficients=array_Degrees_and_clustering_coefficients[:,1]
degree=array_Degrees_and_clustering_coefficients[:,0]


#Find nodes with degree k and calculate their mean
max_k=max(array_Degrees_and_clustering_coefficients[:,0])

meancluster=np.zeros(int(max_k))

for k in np.arange(max_k):
    #Find nodes with degree k
    withdegreek=array_Degrees_and_clustering_coefficients[:,0]==k
    meancluster[int(k)]=np.mean(array_Degrees_and_clustering_coefficients[withdegreek,1])

fig, ax = plt.subplots(1, figsize=(8, 6), sharex=True, sharey=True, squeeze= True)    
ax.scatter(np.arange(max_k),meancluster,color= 'blue', label = 'Numerical calculations',linestyle = '--')
#plt.plot(np.arange(max_k),meancluster,color= 'red', label = 'Numerical calculations',s= 20)
##Theoretical clustering
# k=np.arange(0,50,1)
# E_k = n_q*(k-0.5)-0.5*(n_q)**2
# C_k = 2*E_k/(k*(k-1))
# plt.scatter(k,C_k,color='blue',label = 'Theoretical calculations', alpha=0.5, s= 20) 
plt.ylabel("$C_{k}$", fontsize=12)  
plt.xlabel("Degree $k$", fontsize=12)  
plt.xlim([2,max_k])  # Limiting x axis
plt.rcParams['font.size'] = '20' 
#plt.ylim([min(min(pdf_friendq3),min(P3[N-1,]), min(pdf_friendq8),min(P8[N-1,]),min(pdf_friend50),min(ModifiedFOF_theoretical)),1])  # Limiting y axis
#plt.ylim([0.0000001,max_freq])  # Limiting x axis
#plt.savefig('LambiotteClustering3.pdf')