#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 14:41:27 2023

@author: watsonlevens
"""
import pickle 
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np


def Friend_of_a_friend_model(n_q=1,q=1,N=30): 
    '''
    n_q : Number of neighbours
    q : Probability of a new node to attach to neighbouring nodes
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
    node_size =  [v * 100 for v in degCent.values()]
    plt.figure(figsize=(10,10))
    nx.draw_networkx(G, pos=pos, with_labels=False,
                     node_color=node_color,
                     node_size=node_size )
    #plt.savefig('FOFModel_p0_Visualization.pdf')   # Saving figure  
    plt.axis('off')

    return G  


#G1 = initial_Friend_of_a_friend_model(Gi,m0)
#Network = Friend_of_a_friend_model(G1,N,m0) 
#Network
# Drawing distributions of the graph
######################################################################################################################
def get_all_degree(G):
    #all_degrees = nx.degree(G) # All the node degrees # Undirected network   
    all_in_degrees= G.degree() #All the in_degrees  # Directed network
    #all_degrees = nx.degree(G).values()  # All the degrees
    all_deg= list(nod_degres for nod, nod_degres in all_in_degrees)
    print('mean degree', np.mean(all_deg))
    all_deg=np.array(all_deg).compress((np.array(all_deg)>min(all_deg)).flat)# Array of all degrees with no zero element
    return all_deg


# Making a list of degree values for several graphs and draw the graph that take the average of all degrees. 
all_all_deg=[]  
for rep in range(10): # Reproduce the graph several times and draw the mean of data 
    print('Run number: ',rep)
    # Gi=nx.empty_graph(create_using=nx.Graph())  # Initial graph
    # G1 = initial_Friend_of_a_friend_model(Gi,m0) # Initial graph for friend of friend model 
    nq=1
    n_q=nq+2
    Friendship_Network=Friend_of_a_friend_model(n_q=n_q,q=1,N=10000)
    all_deg=get_all_degree(Friendship_Network)
    all_all_deg.extend(all_deg) 

# ###Saving all_all_deg    
# fileName   =  'all_all_deg_modifiedFOF_n_q50' + ".pkl";
# file_pi = open(fileName, 'wb') 
# pickle.dump(all_all_deg, file_pi)
#
#### PROCCESSING NUMERICAL DATA FOR FRIEND OF A FRIEND MODEL
#
#openfile = open('all_all_deg_modifiedFOF_n_q5.pkl', 'rb') 
#all_all_deg = pickle.load(openfile)    

##Binning
degrees_friend=np.ndarray.flatten(np.array(all_all_deg))  # To make a flat list(array) out of a list(array) of lists(arrays)
xmin =n_q+2 
mybins_friend= np.unique(np.logspace(np.log10(xmin),np.log10(np.max(degrees_friend)+1), num=8, endpoint=True, base=10.0, dtype=int))
degrees_friend=np.ndarray.flatten(np.array(degrees_friend))  # To make a flat list(array) out of a list(array) of lists(arrays)
hist_friend = np.histogram(degrees_friend, bins=mybins_friend)
pdf_friend =hist_friend[0]/np.sum(hist_friend[0])#normalize histogram --for pdf
box_sizes=mybins_friend[1:]-mybins_friend[:-1]  # size of boxes
pdf_friend = pdf_friend/box_sizes   # Divide pdf by its box size


#bins = Midpoints of distribution
mid_points_friend=np.power(10, np.log10(mybins_friend[:-1]) + (np.log10(mybins_friend[1:]-1)-np.log10(mybins_friend[:-1]))/2)


fig, ax = plt.subplots(1, figsize=(10, 6), sharex=True, sharey=True, squeeze= True)
ax.loglog(mid_points_friend,pdf_friend,color= 'red', label="FOF model",linewidth=2)# Watson_David model
plt.ylabel("P(k)", fontsize=12)  
plt.xlabel("k", fontsize=12)  
plt.legend(loc="upper right")
plt.xlim([n_q+2 ,10000])  # Limiting x axis
plt.ylim([0.00000001,1])  # Limiting x axis
#plt.savefig('Modified_FOF_loglog_plot_n_q40.pdf')
     

##### Alpha calculations
###########################################################################################################################
###openfile = open('all_all_deg_a.pkl', 'rb') # N = 100000 for 1000 runs
###all_all_deg = pickle.load(openfile)          # p = 0, q = 1
###############################################################################################################################
#all_all_deg = np.array(all_all_deg)
#x_min = 10#min(all_all_deg) 
#some_deg = all_all_deg[all_all_deg>=x_min]    ##remove zeros from np.array      
#n = len(np.array(some_deg))
#alpha = 1 + n/sum(np.log(some_deg/(x_min-1/2))) ##calculate alpha
#print('alpha:',alpha)      
