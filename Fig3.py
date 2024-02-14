"""
Created on Thu Mar 25 11:17:13 2021

@author: watsonlevens
"""
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
def number_of_triangles(G,target_node):
    nbrs = [nbr for nbr in G.neighbors(target_node)]  
    triangles=0
   
    for j,u in enumerate(nbrs): 
        for v in nbrs[j+1:]:
            if (u in G[v]):
                triangles += 1

    k=len(nbrs) 
    Numerical_Ev=triangles  
    E_v =n_q*(n_q-1)/2 + n_q*(k-n_q)
    return [Numerical_Ev, E_v,k] 


def timestep_triangles(n_q,q,N): 
    
    Numerica_theoretical_Triangle_k= [] ## List of triangles
    '''
    n_q : Number of neighbours
    q : Probability of a new node to attach to neighbouring nodes
    N : Network size
    '''  
    m0= n_q + 2 
    G = nx.complete_graph(m0)
    for source in range(m0, N):                                                            
        existing_nodes = [nod for nod in G.nodes()]
        target_node = random.choice(existing_nodes)
        #target_node = 0
        neighbours = [nbr for nbr in G.neighbors(target_node)]     
       
        G.add_node(source) 
        G.add_edge(source, target_node)                                                                             
        num_neighbours = random.sample(neighbours, min(n_q, len(neighbours))) 
        for neighbor in num_neighbours:
            if random.random() <=q:
                G.add_edge(source, neighbor)
        Numerica_theoretical_Triangle_k.append(number_of_triangles(G,target_node))
    return Numerica_theoretical_Triangle_k 


## Runing the functions
n_q=2
q=1
N=1000

Data = []
for rep in range(100):
    data= timestep_triangles(n_q,q,N)
    Data.append(data)
flattened_list = list(chain(*Data)) # Flattened list of list
Numerical_Ev, Theoretical_Ev,k = zip(*flattened_list) # Unpack data
# Plotting
# Extract unique k and calculate the mean of the corresponding numerical_Ev and theoretical_Ev
unique_degrees = np.unique([entry[2] for entry in flattened_list])
mean_number_of_Numerical_Ev = [np.mean([entry[0] for entry in flattened_list if entry[2] == value]) for value in unique_degrees]
mean_number_of_Theoretical_Ev = [np.mean([entry[1] for entry in flattened_list if entry[2] == value]) for value in unique_degrees]
plt.plot(unique_degrees,mean_number_of_Theoretical_Ev, linestyle='solid',color='red',linewidth=0.5, alpha=1)
plt.plot(unique_degrees, mean_number_of_Numerical_Ev, linestyle='solid', color='blue', linewidth=0.5)
plt.xlabel('Degree $k$')
plt.ylabel('Number of triangles')
#plt.savefig('fof_number_of_triangles_k.pdf')
plt.show()

