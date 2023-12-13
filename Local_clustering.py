"""
Created on Thu Mar 25 11:17:13 2021

@author: watsonlevens
"""
import networkx as nx
import random
import matplotlib.pyplot as plt

def Friend_of_a_friend_model(n_q,q,N): 
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
        local_clustering(G,target_node)
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
    

    E_v =n_q*(n_q-1)/2 + n_q*(k-n_q)    # Tiffany
    
    print('Theoretical E_v:', int(E_v))
    print('Tthoeretical-Numerical E_v:',int(E_v)-triangles)
    
   
    # print('local clustering:',clustering)
    return clustering 
N = 50 # Network size # population size
n_p = 1 # Number of parent nodes
n_q = 5 # Number of neighbours
q = 1 # Probability of a new node to attach to neighbouring nodes
Network = Friend_of_a_friend_model(n_q,q,N)


