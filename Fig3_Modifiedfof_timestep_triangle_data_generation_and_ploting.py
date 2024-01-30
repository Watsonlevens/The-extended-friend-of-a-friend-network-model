"""
Created on Thu Mar 25 11:17:13 2021

@author: watsonlevens
"""
import pickle 
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np

def number_of_triangles(G,target_node):
    nbrs = [nbr for nbr in G.neighbors(target_node)]  
    triangles=0
   
    for j,u in enumerate(nbrs):
        # Find all pairs who are nbrs of a node. 
        for v in nbrs[j+1:]:
            if (u in G[v]):
                triangles += 1

    k=G.degree(target_node) # Degree of a target node
    Numerical_Ev=triangles  # Number of numerical triangles
    E_v =n_q*(n_q-1)/2 + n_q*(k-n_q)    ## Number of Triangles from theoretical equation derived by Tiffany
    print('\n')
    print('Theoretical:',E_v)
    print('Numerical_Ev:', Numerical_Ev)
    return [Numerical_Ev, E_v] 


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
        #target_node = 0

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
        Numerica_theoretical_Triangle.append(number_of_triangles(G,target_node))
    return Numerica_theoretical_Triangle 
   
  


## Runing the functions
n_q=2
q=1
N=100

data_random = timestep_triangles(n_q,q,N)

# # # Saving the data
# # #######################################################################################################################
# fileName   =  'EV_worstcase_triangles_random_node' + ".pkl";
# file_pi = open(fileName, 'wb') 
# pickle.dump(data, file_pi)
# # #######################################################################################################################

Numerical_Evrandom, Theoretical_Evrandom = zip(*data_random) # Unpack data
#########################################################################################################################
t=np.arange(len(Numerical_Evrandom)) # time steps same as np.arange(N) where N is the network size

#Plot
##fig,(ax1,ax2)=plt.subplots(nrows=1, ncols=2,figsize=(18, 6), sharex=False, sharey=False, squeeze= True)
fig,ax=plt.subplots(nrows=1, ncols=1,figsize=(18, 6), sharex=False, sharey=False, squeeze= True)
# Set general font size
plt.rcParams['font.size'] = '20'
# Differentiate some x-axis points with dots of unique colors
highlight_indices = np.arange(1,90,5)  # You can change these indices as needed

ax.plot(t, Numerical_Evrandom,color='blue',label = 'Numerical simulation', alpha=1, linestyle ='--',linewidth=2)
ax.plot(t,Theoretical_Evrandom,color='red',label = 'Theoretical calculations', alpha=1)

for ax in fig.get_axes():
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
 	    label.set_fontsize(20)
    ax.set(xlabel='Time step $t$', ylabel='Number of triangles')
    ax.label_outer()  # Set axis scales outer
# ### Setting the limits of the plot
ax.axes.set_xlim([0,N])
ax.axes.set_ylim([min(Numerical_Evrandom),max(Numerical_Evrandom)+10])