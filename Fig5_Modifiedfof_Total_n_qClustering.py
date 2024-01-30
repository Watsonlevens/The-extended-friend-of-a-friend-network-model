#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 22:45:15 2021

@author: watsonlevens
"""

from networkx.utils import not_implemented_for
from networkx.utils import py_random_state
import pickle 
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np


def Friend_of_a_friend_model(n_q,q,N): 
    '''
    n_q : Number of neighbours
    q : Probability of a new node to attach to neighbouring nodes
    N : Network size
    '''  
    # Initializing the network
    m0= n_q + 2 # Initial number of nodes
    G = nx.complete_graph(m0)  # Initial graph
    
            # Growth of the network
    for source in range(m0, N): # Start connection from m0 node and stop at N
        #################################################################################################################                    
        # Step1. Pick one node randomly and make connection.
        # ########################################################################################################################                                                           
        existing_nodes = [nod for nod in G.nodes()]
        target_node = random.choice(existing_nodes)

        neighbours = [nbr for nbr in G.neighbors(target_node)]     
       
        G.add_node(source) ## Adding node to the network # SOURCE MUST BE ADDED AFTER RANDOM SAMPLING SO AS 
                            ## TO AVOID SELF LOOP WHEN ADDING EDGES
        G.add_edge(source, target_node) # Add edge 
    
        ################################################################################################################                    
        ##### Step2. Independently link the target node with each of n_q neighbors with probability q
                     # Pick n_q nodes randomly without replacement and with prob q, make connection.
        ########################################################################################################################                                                                               
        num_neighbours = random.sample(neighbours, min(n_q, len(neighbours))) # Random neighbors of the target node
        for neighbor in num_neighbours:
            if random.random() <=q:
                G.add_edge(source, neighbor)
        
    return G 


#Numerical calculations of clustering
################################################################################ 
@py_random_state(2)
@not_implemented_for('directed') 
def node_clustering(G, trials=1000, seed=None):
    #Calculates  [trials] weakly connected triangle clusters in graph G
    n = len(G)
    #triangles = 0
    nodes = list(G)
    
    #List of all degree, triangle pairs
    degree_and_triangles=[]
    
    
    for i in [int(seed.random() * n) for i in range(trials)]:
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
           #print(triangles)
           degree_and_triangles.append([degree,(triangles/(degree*(degree-1)/2))])  
        else:
             degree_and_triangles.append([degree,0])
    return degree_and_triangles 




N=1000
n_qs= np.arange(1,40)
q = 1

Total_C_k= []
for n_q in n_qs: 
    G=Friend_of_a_friend_model(n_q,q,N)
    numtrials=1000
    Degrees_and_clustering_coefficients =node_clustering(G,trials=numtrials)
    array_Degrees_and_clustering_coefficients=np.array(Degrees_and_clustering_coefficients)
    clustering_coefficients=array_Degrees_and_clustering_coefficients[:,1]
    degree=array_Degrees_and_clustering_coefficients[:,0]
    
    
    #Find nodes with degree k and calculate their mean
    max_k=max(array_Degrees_and_clustering_coefficients[:,0])
    
    
    meancluster=np.zeros(int(max_k))  # numpy mean clusters
    propcluster=np.zeros(int(max_k)) # numpy proportional of clusters            
    for k_i in np.arange(max_k):
        
        #Find nodes with degree k
        withdegreek_i=array_Degrees_and_clustering_coefficients[:,0]==k_i
        # Computer mean cluster of nodes with degree k
        meancluster[int(k_i)]=np.mean(array_Degrees_and_clustering_coefficients[withdegreek_i,1])
        # Compute proportional of clusters for nodes with degree k.
        propcluster[int(k_i)]=np.sum(withdegreek_i)/numtrials
    ## Replacing NaN with 0
    meancluster=np.nan_to_num(meancluster)

    #Frequency of numerical clusters with degree k.
# Normalize with sum of prportional of nodes with k>=n_q+2 (i.e sum(propcluster[n_q+2:]))
    clustering=sum(meancluster[n_q+2:]*propcluster[n_q+2:])/sum(propcluster[n_q+2:])
    #print('Global clustering',clustering)
    Total_C_k.append(clustering)

## Saving the clusterings
########################################################################################################################
#fileName   =  'k_NumeriClustering_b' + ".pkl";
#file_pi = open(fileName, 'wb') 
#pickle.dump(meancluster, file_pi)
########################################################################################################################
##
#Plot
fig,ax1=plt.subplots(1,1,figsize=(18, 8), sharey=True, squeeze= True)
# Set general font size
plt.rcParams['font.size'] = '20'
 #Plot
 
 #p-clustering plots
ax1.scatter(n_qs,Total_C_k,color='blue',label = 'Numerical simulation', alpha=0.5,marker="s", s = 25)
# ax1.plot(np.sort(prob),np.sort(p_Theorclusters_a),color='red',label = 'Theoretical calculations', alpha=0.5)

# ax2.scatter(prob,q_NumeriClustering_a,color='blue',label = 'Numerical simulation', alpha=0.5,marker="s", s = 25)
# ax2.plot(np.sort(prob),np.sort(q_Theorclusters_a),color='red',label = 'Theoretical calculations', alpha=0.5)
    
ax1.set(xlabel='Number of neighbours $n_q$', ylabel='$C_T$')
# ax2.set(xlabel='Probability $p$', ylabel='$C_T$')








