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
    


### The function for for simulating the clustering coefficient
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
        #Can take same node twice or more. Can be a problem on small graphs.
        nbrs = list(G[nodes[i]])
        degree=len(nbrs)
        
        triangles=0
        
#            
        for j,u in enumerate(nbrs):
            #Find all pairs
            for v in nbrs[j+1:]:
                if (u in G[v]):
                    triangles += 1
        if degree > 1:
            degree_and_triangles.append([degree,(triangles/(degree*(degree-1)/2))])  
        else:
            degree_and_triangles.append([degree,0])
            
    return degree_and_triangles 



## Using the functions
n_q=2
q=1
N=1000
Network=  Friend_of_a_friend_model(n_q,q,N)
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
    



fig, ax = plt.subplots(1, figsize=(10, 8), sharex=True, sharey=True, squeeze= True)
ax.scatter(np.arange(max_k),meancluster,color= 'blue', label = 'Numerical calculations',s= 20)

##Theoretical clustering
# k=np.arange(0,50,1)
# E_v =n_q*(n_q-1)/2 + n_q*(k-n_q)    
# C_v =2*n_q/N
# C_k = 2*E_v/(k*(k-1))


# plt.scatter(k,C_k,color='blue',label = 'Theoretical calculations', alpha=0.5, s= 20) 
plt.ylabel("$C_{k}$", fontsize=12)  
plt.xlabel("Degree $k$", fontsize=12) 
plt.rcParams['font.size'] = '20' 
#plt.legend(loc="upper right")
#plt.savefig('ModifiedClustering.pdf')
