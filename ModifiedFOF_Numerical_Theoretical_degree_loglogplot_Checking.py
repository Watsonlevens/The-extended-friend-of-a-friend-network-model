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

def displayfriendoffriend(G):
#    nx.draw_networkx(G_dispay, node_size=40, node_color='red', pos=nx.spring_layout(G_dispay),with_labels = False)
#    #########plt.savefig('Network_Display_N' + str(n) + '.pdf')   # Saving figure  
#    plt.show()
    
    pos = nx.spring_layout(G)
    degCent = nx.degree_centrality(G)
    #eigenCent =nx.eigenvector_centrality(Gi)
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

######################################################################################################################
def get_all_degree(G):
    all_degrees= G.degree() #All degrees
    all_deg= list(nod_degres for nod, nod_degres in all_degrees)
    #print('mean degree', np.mean(all_deg))
    #all_deg=np.array(all_deg).compress((np.array(all_deg)>min(all_deg)).flat)# Array of all degrees with no zero element
    return all_deg


def degrees_and_numericalP_k (n_q,q,N,n): ## Degrees and PDF for the numerical simulations
    '''
    Parameters
    n_q : Number of neighbours
    q : Probability of a new node to attach to neighbouring nodes
    N : Network size
    n : number of realization
    '''
    all_all_deg=[]  # Making a list of degree values for several graphs
    for rep in range(n): # Reproduce the graph several times 
        print('Run number: ',rep) 
        G=Friend_of_a_friend_model(n_q,q,N) 
        all_deg=get_all_degree(G)
        all_all_deg.extend(all_deg) 
    '''
    Binning degrees and calculation of PDF
    '''
    degrees_friend=np.ndarray.flatten(np.array(all_all_deg))  # To make a flat list(array) out of a list(array) of lists(arrays)
    xmin =2 + n_q 
    mybins_friend= np.unique(np.logspace(np.log10(xmin),np.log10(np.max(degrees_friend)+1), num=8, endpoint=True, base=10.0, dtype=int))
    degrees_friend=np.ndarray.flatten(np.array(degrees_friend))  # To make a flat list(array) out of a list(array) of lists(arrays)
    hist_friend = np.histogram(degrees_friend, bins=mybins_friend)
    pdf_friend =hist_friend[0]/np.sum(hist_friend[0])#normalize histogram --for pdf
    box_sizes=mybins_friend[1:]-mybins_friend[:-1]  # size of boxes
    pdf_friend = pdf_friend/box_sizes   # Divide pdf by its box size
    
    #bins = Midpoints of distribution
    mid_points_friend=np.power(10, np.log10(mybins_friend[:-1]) + (np.log10(mybins_friend[1:]-1)-np.log10(mybins_friend[:-1]))/2)
    degrees =   mid_points_friend
    return [int(i) for i in degrees], pdf_friend




def theoreticalfof(n_q ,q, max_k):  ## Degrees and PDF for the theoretical calculations
    '''
    max_k : maximum degree value for n realization of the fof function)
    n_q : Number of neighbours
    q : Probability of a new node to attach to neighbouring nodes
    '''
    k=np.arange(2 + n_q,max_k) 
    a = 2*(1+(q*n_q))
    b = a/(q*n_q)
    
    l =b*(a+b)**b 
    P_k=l*(k+b)**(-(1+b))
    
    
    ## Combined Eqn, equation a whole
    #P_k=((2*(1+(q*n_q))/(q*n_q))*((2*(1+(q*n_q)))+(2*(1+(q*n_q))/(q*n_q)))**(2*(1+(q*n_q))/(q*n_q)))*((k+(2*(1+(q*n_q))/(q*n_q)))**(-(1+(2*(1+(q*n_q)))/(q*n_q))))
    return k, P_k 





### Usage of the functions
#N=1514435
N=1000000
n_q=3
q = 1
n=1
kn_and_Pkn= degrees_and_numericalP_k (n_q,q,N,n)
max_k = max(kn_and_Pkn[0])
k_numerical=kn_and_Pkn[0]
Pk_numeical=kn_and_Pkn[1]

kt_and_Pkt = theoreticalfof(n_q ,q, max_k)
k_theoretical= kt_and_Pkt[0]
Pk_theoretical=kt_and_Pkt[1]

fig, ax = plt.subplots(1, figsize=(10, 6), sharex=True, sharey=True, squeeze= True)
ax.loglog(k_numerical,Pk_numeical,color= 'red', label="Numerical",linewidth=2)# Watson_David model
ax.loglog(k_theoretical,Pk_theoretical, alpha=1, label = 'Mean-field approach')

plt.ylabel("P(k)", fontsize=12)  
plt.xlabel("k", fontsize=12)  
plt.legend(loc="upper right")
# plt.xlim([20 ,1000])  # Limiting x axis
# #plt.xlim([2 + n_q ,1000])  # Limiting x axis
# plt.ylim([0.00000001,0.1])  # Limiting x axis
#plt.savefig('Modified_FOF_loglog_plot_n_q40.pdf')

#plt.savefig('Modified_FOF_loglog_plot_n_q3.pdf')
 
# ### Saving   
# fileName   =  'NumericalDegree_and_Pk_n_q3' + ".pkl";
# file_pi = open(fileName, 'wb') 
# pickle.dump(kn_and_Pkn, file_pi)

# # #### Loading data
# # openfile = open('NumericalDegree_and_Pk_n_q3.pkl', 'rb') 
# # kn_and_Pkn = pickle.load(openfile)  
    

# ### Saving   
# fileName   =  'TheoreticalDegree_and_Pk_n_q3' + ".pkl";
# file_pi = open(fileName, 'wb') 
# pickle.dump(kt_and_Pkt, file_pi)

# #### Loading data
# openfile = open('TheoreticalDegree_and_Pk_n_q3.pkl', 'rb') 
# kt_and_Pkt = pickle.load(openfile)

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
