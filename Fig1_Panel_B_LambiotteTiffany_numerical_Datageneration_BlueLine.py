#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 11:17:13 2021

@author: watsonlevens
"""
import pickle 
import networkx as nx
import random
N = 1000 # Network size # population size
q = 0.8 # Probability of attaching to neighbour nodes (q must be <1 for ploting degree distribution)
m0 = 1 # Initial node.
Gi=nx.empty_graph(create_using=nx.Graph())  # Initial empty graph
     
# A function to generate Lambiotte model  
def Lambiotte_model(G,N):
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
        while len(neighbours)>0: # n_q = infinity. all neighbours need to make connection with the new node
           nbr = neighbours.pop() #
           if q >random.random():  
              G.add_edge(source, nbr) # Add edge 
    return G 

# Drawing degree distributions of the graph
######################################################################################################################
 # A function to get degree of all nodes 
def get_all_degree(G):
    #all_degrees = nx.degree(G) # All the node degrees # Undirected network   
    all_in_degrees= G.degree() #All the in_degrees  # Directed network
    #all_degrees = nx.degree(G).values()  # All the degrees
    all_deg= list(nod_degres for nod, nod_degres in all_in_degrees)
    #print('mean indegrees', np.mean(all_deg))
    #all_deg=np.array(all_deg).compress((np.array(all_deg)>min(all_deg)).flat)# Array of all degrees with no zero element
    return all_deg


# Making a list of degree values for several graphs and draw the distribution that takes the average of all degrees. 
Lambiotte_all_all_deg=[]  
n=10 # Number of realization 
for rep in range(n): # Reproduce the graph several times and draw the mean of data 
    #print('Run number: ',rep)
    Gi=nx.empty_graph(create_using=nx.Graph())  # Initial empty graph
    Lambiotte_Network=Lambiotte_model(Gi,N)
    all_deg=get_all_degree(Lambiotte_Network)
    Lambiotte_all_all_deg.extend(all_deg) 



## Saving the degrees        
######################################################################################################################
TheorfileName   =  'Lambiotte_all_all_degq8' + ".pkl";
Theofile_pi = open(TheorfileName, 'wb') 
pickle.dump(Lambiotte_all_all_deg, Theofile_pi)
######################################################################################################################



