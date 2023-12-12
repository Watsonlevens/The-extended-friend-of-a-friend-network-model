"""
Created on Thu Mar 25 11:17:13 2021

@author: watsonlevens
"""
import pickle 
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np

def timestep_triangles(n_q,q,N): 
    
    Numerica_theoretical_Triangle= [] ## List of triangles
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
         
        ## Calculate triangles  and append it in the list above
        Numerica_theoretical_Triangle.append(triangles(G,target_node))
    return Numerica_theoretical_Triangle 
   
  
def triangles(G,target_node):
    nbrs = [nbr for nbr in G.neighbors(target_node)]  
    triangles=0
   
    for j,u in enumerate(nbrs):
        # Find all pairs who are nbrs of a node. 
        for v in nbrs[j+1:]:
            if (u in G[v]):
                triangles += 1

    k=G.degree(target_node)
    Numerical_Ev=triangles
    E_v =n_q*(n_q-1)/2 + n_q*(k-n_q)    ## Triangles theoretical equation # Tiffany
    print('\n')
    print('Theoretical:',E_v)
    print('Numerical_Ev:', Numerical_Ev)
    return [Numerical_Ev, E_v] 



## Runing the functions
n_q=2
q=1
N=100

data = timestep_triangles(n_q,q,N)

# # Saving the data
# #######################################################################################################################
fileName   =  'EV_worstcase_triangles_random_node' + ".pkl";
file_pi = open(fileName, 'wb') 
pickle.dump(data, file_pi)
# #######################################################################################################################
# #
Numerical_Ev, Theoretical_Ev = zip(*data) # Unpack data
t=np.arange(len(Numerical_Ev)) # time steps
# Plotting
plt.plot(t,Theoretical_Ev, linestyle='solid',color='red',linewidth=0.5, alpha=1)
plt.plot(t, Numerical_Ev, marker='d', linestyle='--', color='blue', linewidth=1)
plt.xlabel('Time step $t$')
plt.ylabel('Number of triangles')
plt.show()