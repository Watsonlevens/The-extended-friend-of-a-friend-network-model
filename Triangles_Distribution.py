#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 11:17:13 2021

@author: watsonlevens
"""
import networkx as nx
import random
import matplotlib.pyplot as plt
  
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
    # eigenCent =nx.eigenvector_centrality(Gi)
    #betCent = nx.betweenness_centrality(Gi, normalized=True, endpoints=True)
   
    node_color = [20000.0 * G.degree(v) for v in G]
    node_size =  [v * 1000 for v in degCent.values()]
    plt.figure(figsize=(10,10))
    nx.draw_networkx(G, pos=pos, with_labels=True,
                     node_color=node_color,
                     node_size=node_size )
    #plt.savefig('FOFModel_p0_Visualization.pdf')   # Saving figure  
    plt.axis('off')

    return G 


def plot_triangleshistogram(G,q=0.4,n_q=1,N=30):
    number_triangles = []
    for i in range(100):
        G= Friend_of_a_friend_model(n_q=n_q,q=q,N=N)  
        # Calculate the number of triangles for each node
        # Count triangles in the graph
        trianglesum = sum(nx.triangles(G).values()) # sums up the values in a dictionary. Each triangle is counted as a triangle for each of the three nodes. Thus the sum of the values should be 3 times the number of triangles. 
        num_triang = trianglesum/3  # Gives total triangles in a network
        
        #print('Number of triangles:',num_triang)
        number_triangles.append(num_triang)
    
    
    # Plot the distribution of the number of triangles
    #fig, ax = plt.subplots(1, figsize=(10, 6), sharex=True, sharey=True, squeeze= True)
    plt.hist(number_triangles, bins=10, edgecolor="yellow", color="green")
    plt.xlabel('Number of Triangles')
    plt.ylabel('Count')
    plt.title('Distribution of Triangles in the Network')
    plt.grid(True)
    plt.show()


def predict_probability_q(G,q=0.5,N=30):
    G= Friend_of_a_friend_model(n_q=1,q=q,N=N) 
    trianglesum = sum(nx.triangles(G).values()) # sums up the values in a dictionary. Each triangle is counted as a triangle for each of the three nodes. Thus the sum of the values should be 3 times the number of triangles. 
    num_triang = trianglesum/3  # Gives total triangles in a network     
    predicted_q = num_triang/N
    return predicted_q


def plot_numerical_theoretical_triangles(G,q=1,n_q=1,N=30):
    Theoretical_traingles = []
    number_triangles = []
    #Random probabilities
    #############################################################################################################
    probs =[random.random() for p in range(100)]
    # openfile = open('probs.pkl', 'rb') 
    # prob = pickle.load(openfile)
    for q in probs:
        G = Friend_of_a_friend_model(n_q=n_q,q=q,N=N)
        # Calculate the number of triangles for each node
        # Count triangles in the graph
        trianglesum = sum(nx.triangles(G).values()) # sums up the values in a dictionary. Each triangle is counted as a triangle for each of the three nodes. Thus the sum of the values should be 3 times the number of triangles. 
        num_triang = trianglesum/3  # Gives total triangles in a network
        number_triangles.append(num_triang)
        #N=1000
        Theoretical_traingles.append(q*(N-4)+1)
    
    
    # Plot the distribution of the number of triangles
    fig, ax = plt.subplots(1, figsize=(10, 8), sharex=True, sharey=True, squeeze= True)
    ax.plot(probs,Theoretical_traingles,color= 'red',linestyle ='--', label="Theoretical simulation",linewidth=0.4)
    ax.scatter(probs,number_triangles,color= 'blue', label="Numerical simulation",s=10, marker='o',)
    plt.xlabel('q')
    plt.ylabel('Number of triangles')
    plt.rcParams['font.size'] = '20'
    #plt.savefig('LambiotteModifiedfof_traingles_Vs_q.pdf')
    #plt.show()




## Usage of the functions
G= Friend_of_a_friend_model(n_q=1,q=1,N=50) 
# displayfriendoffriend(G)
# plot_triangleshistogram(G,q=0.5,N=100)
# predict_probability_q(G,q=0.5,N=500)
plot_numerical_theoretical_triangles(G,q=1,n_q=1,N=1000)
