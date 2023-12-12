#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 11:17:13 2021

@author: watsonlevens
"""
import pickle 
import networkx as nx
import random
import math
import scipy
from decimal import Decimal
from scipy import special
import scipy.special as sc
import matplotlib.pyplot as plt
import numpy as np
N = 1000 # Network size # population size
q = 0.3 # Probability of attaching to neighbour nodes (q must be <1 for ploting degree distribution)
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
        while len(neighbours)>0: # all neighbours need to make connection with the new node
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
all_all_deg=[]  
for rep in range(1): # Reproduce the graph several times and draw the mean of data 
    #print('Run number: ',rep)
    Gi=nx.empty_graph(create_using=nx.Graph())  # Initial empty graph
    Lambiotte_Network=Lambiotte_model(Gi,N)
    all_deg=get_all_degree(Lambiotte_Network)
    all_all_deg.extend(all_deg) 


###PROCCESSING NUMERICAL DEGREES FOR THE MODEL   
#Binning of degree
degrees_friend=np.ndarray.flatten(np.array(all_all_deg))  # To make a flat list(array) out of a list(array) of lists(arrays)
xmin =min(all_all_deg)
mybins_friend= np.unique(np.logspace(np.log10(xmin),np.log10(np.max(degrees_friend)+1), num=25, endpoint=True, base=10.0, dtype=int))
degrees_friend=np.ndarray.flatten(np.array(degrees_friend))  # To make a flat list(array) out of a list(array) of lists(arrays)

hist_friend = np.histogram(degrees_friend, bins=mybins_friend) # Histogram of the data
pdf_friend1 =hist_friend[0]/np.sum(hist_friend[0])#normalize histogram --for pdf


box_sizes=mybins_friend[1:]-mybins_friend[:-1]  # size of boxes
pdf_friend = pdf_friend1/box_sizes   # Divide pdf by its box size

#bins = Midpoints of distribution
mid_points_friend=np.power(10, np.log10(mybins_friend[:-1]) + (np.log10(mybins_friend[1:]-1)-np.log10(mybins_friend[:-1]))/2)


### TIFFANY SOLUTION FOR LAMBIOTTE MODEL
P = np.zeros(shape=(N,N))  # expected degree distribution at time j
P[0,0] = 1

## Identity matrix
## row/column j corresponds to vertices of degree j+1
I = np.zeros(shape=(N,N))
for i in range(N):
    I[i,i] = 1

Q = np.zeros(shape=(N,N))
for j in range(N):
    Q[j,j]= -(2+(j+1)*(q-(1-q)*q**(j)))

for j in range(N-1):
    Q[j,j+1]=1+q**(j+1) + (q*(j+1))   
    
for k in range(N):
  for j in range(k):
        #bionomial combination
    bioexp = Decimal((q**(j))*(1-q)**(k+1-j))
    Q[k,j] = sc.comb(k+1, j,exact=True)*bioexp

# Solve for the n-th row of P iteratively, using 
# the matrix equation in the document
k = np.arange(N)
for j in range(1,N):
    #print('j:',j)
    A=np.zeros(shape=(N,N))
    A[0:j,]=Q[0:j,]/(j+1)
    update=np.dot((P[j-1,]).transpose(),(I+A))
    P[j,]=update


## Ploting the figure
fig, ax = plt.subplots(1, figsize=(10, 6), sharex=True, sharey=True, squeeze= True)
ax.loglog(k,P[N-1,],color= 'red', label="Tiffany solution", linewidth=2)# P[N-1,] is a row N-1 of matrix P 
ax.loglog(mid_points_friend,pdf_friend,color= 'blue',linestyle ='--', label="Lambiotte simulation",linewidth=2)

plt.ylabel("P(k)", fontsize=12)  
plt.xlabel("k", fontsize=12)  
#plt.legend(loc="upper right")
plt.xlim([1,1000])  # Limiting x axis
plt.ylim([min(pdf_friend),1])  # Limiting x axis
#plt.savefig('LambiotteTiffany_loglog_plot2.pdf')