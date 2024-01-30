"""
Created on Wed Aug 25 10:53:09 2021
@author: watsonlevens
"""
import pickle 
from decimal import Decimal
import scipy.special as sc
import numpy as np

N = 1000 # Network size # population size
q = 0.3 # Probability of attaching to neighbour nodes (q must be <1 for ploting degree distribution)
m0 = 1 # Initial node.

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




# Saving expected degree distribution
       
######################################################################################################################
TheorfileName   =  'Theoretical_Tiffanyq3' + ".pkl";
Theofile_pi = open(TheorfileName, 'wb') 
pickle.dump(P, Theofile_pi)
######################################################################################################################


# ## Ploting the figure
# fig, ax = plt.subplots(1, figsize=(10, 6), sharex=True, sharey=True, squeeze= True)
# ax.loglog(k,P[N-1,],color= 'red', label="Tiffany solution", linewidth=2)# P[N-1,] is a row N-1 of matrix P 

# plt.ylabel("P(k)", fontsize=12)  
# plt.xlabel("k", fontsize=12)  
# #plt.legend(loc="upper right")
# plt.xlim([1,1000])  # Limiting x axis
# #plt.savefig('LambiotteTiffany_loglog_plot2.pdf')













  








