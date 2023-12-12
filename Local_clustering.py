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

N = 50 # Network size # population size
n_p = 1 # Number of parent nodes
n_q = 5 # Number of neighbours
q = 1 # Probability of a new node to attach to neighbouring nodes
m0 = n_p + n_q + 1 # Initial number of edges
Gi=nx.empty_graph(create_using=nx.Graph())  # Initial empty graph
# A function returning an initial friend of friend network
def initial_Friend_of_a_friend_model(Gi,m0):   #Initial graph
    Initial_Nodes = np.random.permutation(m0) # List of initial nodes 
    for node1 in Initial_Nodes: # Make n_p + n_q connections
        for node2 in Initial_Nodes:
            if node1 != node2: # Avoid self loop
                Gi.add_edge(node1, node2) # Add edge 
    return Gi
     
   
def Friend_of_a_friend_model(G,N,m0):                
    for source in range(m0, N): # Start connection from m0 node and stop at N
        #################################################################################################################                    
        # Step1. Pick one node randomly and make connection.
        # ########################################################################################################################                                                           
        nodes = [nod for nod in G.nodes()]
        #node = random.choice(nodes)
        node = 0

        neighbours = [nbr for nbr in G.neighbors(node) # neighborhoods are nodes followed by target
                                    if not G.has_edge(source, nbr) # If no edge btn source node and nbr node (followed node)
                                    and not nbr == source] # If neighbor node is not equal to source node       
        G.add_node(source) ## Adding node to the network # SOURCE MUST BE ADDED AFTER RANDOM SAMPLING SO AS 
                           ## TO AVOID SELF LOOP WHEN ADDING EDGES
        G.add_edge(source, node) # Add edge 
    
        #################################################################################################################                    
        # Step2. Pick n_q nodes randomly and with prob q, make connection.
        # ########################################################################################################################                                                                               
        num_nbrs =0
        while num_nbrs<n_q and len(neighbours)>0: 
            nbr = neighbours.pop(random.randrange(len(neighbours))) # Choose randomly among the many nbrs available
            if q >random.random():  
               G.add_edge(source, nbr) # Add edge 
               num_nbrs = num_nbrs + 1
        local_clustering(G,node)
    return G 
    
####  Display the graph    
def displayfriendoffriend(G):
#    nx.draw_networkx(G_dispay, node_size=40, node_color='red', pos=nx.spring_layout(G_dispay),with_labels = False)
#    #########plt.savefig('Network_Display_N' + str(n) + '.pdf')   # Saving figure  
#    plt.show()
    
    pos = nx.spring_layout(G)
    degCent = nx.degree_centrality(G)
    #    eigenCent =nx.eigenvector_centrality(Gi)
    #betCent = nx.betweenness_centrality(Gi, normalized=True, endpoints=True)
    node_color = [20000.0 * G.degree(v) for v in G]
    node_size =  [v * 2000 for v in degCent.values()]
    plt.figure(figsize=(10,10))
    nx.draw_networkx(G, pos=pos, with_labels=True,
                     node_color=node_color,
                     node_size=node_size )
    #plt.savefig('FOFModel_p0_Visualization.pdf')   # Saving figure  
    plt.axis('off')

    return G 


def local_clustering(G,node):
    nbrs = [nbr for nbr in G.neighbors(node)]  
    triangles=0
   
    for j,u in enumerate(nbrs):
        # Find all pairs who are nbrs of a node. 
        for v in nbrs[j+1:]:
            if (u in G[v]):
                triangles += 1
    clustering = triangles/(G.degree(node)*(G.degree(node)-1)/2)
    displayfriendoffriend(G)
    
    # E_v for nq=1 and nq = 2
    print('\n')
    print('node:',node)
    k=G.degree(node)
    print('node degree, k :',k)
    print('triangles(E_v):',triangles)
    print('E_v = k + ',triangles-k)
    
    #E_v = n_q*(k-2)+1  # Watson
    E_v =n_q*(n_q-1)/2 + n_q*(k-n_q)    # Tiffany
    
    print('Theoretical E_v:', int(E_v))
    print('Tthoeretical-Numerical E_v:',int(E_v)-triangles)
    
    # ## General E_v for any nq
    # print('\n')
    # print('node:',node)
    # k=G.degree(node)
    # print('node degree, k :',k)
    # print('triangles(E_k):',triangles)
    # print('E_v = k + ',triangles-k)
    
    # E_v = n_q*(k-1/2)-1/2*(n_q)**2  # Watson
    #E_v =n_q*(n_q-1)/2 + n_q*(k-n_q)    # Tiffany
    # print('Theoretical E_v:', int(E_v))
    # print('Thoeretical-Numerical E_v:',int(E_v)-triangles)

    
   
    # print('local clustering:',clustering)
    return clustering 
G = initial_Friend_of_a_friend_model(Gi,m0)
Network = Friend_of_a_friend_model(G,N,m0)
# node = 0
# print(local_clustering(Network,node))
