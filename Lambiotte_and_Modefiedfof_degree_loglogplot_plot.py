#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 21:25:56 2021

@author: watsonlevens
"""
import pickle 
import matplotlib.pyplot as plt
import numpy as np

# Lambiotte theoretical results for n_q = infinity, q = 0.3
#########################################################################################################################
openfile = open('Theoretical_Tiffanyq3.pkl', 'rb') 
P3 = pickle.load(openfile)
#########################################################################################################################
# Lambiotte theoretical results n_q = infinity, q=0.8
#########################################################################################################################
openfile = open('Theoretical_Tiffanyq8.pkl', 'rb') 
P8 = pickle.load(openfile)
#########################################################################################################################

# # Lambiotte numerical results for n_q = infinity, q = 0.3
#########################################################################################################################
openfile = open('Lambiotte_all_all_degq3.pkl', 'rb') 
all_all_degq3 = pickle.load(openfile)
#########################################################################################################################
##########################################################################################################################
# Lambiotte numerical results for n_q = infinity, q = 0.8
########################################################################################################################
openfile = open('Lambiotte_all_all_degq8.pkl', 'rb') 
all_all_degq8 = pickle.load(openfile)
#########################################################################################################################


# # Modefied numerical results for n_q = infinity, q = 0.3
#########################################################################################################################
openfile = open('ModifiedFOF_theoretical_n_q40.pkl', 'rb') 
ModifiedFOF_theoretical = pickle.load(openfile)
#########################################################################################################################
##########################################################################################################################
# Modefied theoretical results for n_q = infinity, q = 0.8
########################################################################################################################
openfile = open('ModifiedFOF_all_all_deg_n_q40.pkl', 'rb') 
ModifiedFOF_numerical = pickle.load(openfile)
#########################################################################################################################


# ### Saving   
# fileName   =  'NumericalDegree_and_Pk' + ".pkl";
# file_pi = open(fileName, 'wb') 
# pickle.dump(kn_and_Pkn, file_pi)

#### Loading data
openfile = open('NumericalDegree_and_Pk.pkl', 'rb') 
kn_and_Pkn = pickle.load(openfile) 

#### Loading data
openfile = open('TheoreticalDegree_and_Pk.pkl', 'rb') 
kt_and_Pkt = pickle.load(openfile) 

###PROCCESSING NUMERICAL DEGREES FOR THE MODEL   
#Binning for q = 0.3
degrees_friendq3=np.ndarray.flatten(np.array(all_all_degq3))  # To make a flat list(array) out of a list(array) of lists(arrays)
xmin =min(all_all_degq3)
mybins_friendq3= np.unique(np.logspace(np.log10(xmin),np.log10(np.max(degrees_friendq3)+1), num=25, endpoint=True, base=10.0, dtype=int))
degrees_friendq3=np.ndarray.flatten(np.array(degrees_friendq3))  # To make a flat list(array) out of a list(array) of lists(arrays)

hist_friendq3 = np.histogram(degrees_friendq3, bins=mybins_friendq3) # Histogram of the data
pdf_friendq3 =hist_friendq3[0]/np.sum(hist_friendq3[0])#normalize histogram --for pdf


box_sizes=mybins_friendq3[1:]-mybins_friendq3[:-1]  # size of boxes
pdf_friendq3 = pdf_friendq3/box_sizes   # Divide pdf by its box size

#bins = Midpoints of distribution
mid_points_friendq3=np.power(10, np.log10(mybins_friendq3[:-1]) + (np.log10(mybins_friendq3[1:]-1)-np.log10(mybins_friendq3[:-1]))/2)




#Binning for q = 0.8
degrees_friendq8=np.ndarray.flatten(np.array(all_all_degq8))  # To make a flat list(array) out of a list(array) of lists(arrays)
xmin =min(all_all_degq8)
mybins_friendq8= np.unique(np.logspace(np.log10(xmin),np.log10(np.max(degrees_friendq8)+1), num=25, endpoint=True, base=10.0, dtype=int))
degrees_friendq8=np.ndarray.flatten(np.array(degrees_friendq8))  # To make a flat list(array) out of a list(array) of lists(arrays)

hist_friendq8 = np.histogram(degrees_friendq8, bins=mybins_friendq8) # Histogram of the data
pdf_friendq8 =hist_friendq8[0]/np.sum(hist_friendq8[0])#normalize histogram --for pdf


box_sizes=mybins_friendq8[1:]-mybins_friendq8[:-1]  # size of boxes
pdf_friendq8 = pdf_friendq8/box_sizes   # Divide pdf by its box size

#bins = Midpoints of distribution
mid_points_friendq8=np.power(10, np.log10(mybins_friendq8[:-1]) + (np.log10(mybins_friendq8[1:]-1)-np.log10(mybins_friendq8[:-1]))/2)



#Binning for modified friend of friend model
degrees_friend50=np.ndarray.flatten(np.array(ModifiedFOF_numerical))  # To make a flat list(array) out of a list(array) of lists(arrays)
xmin =min(ModifiedFOF_numerical)
mybins_friend50= np.unique(np.logspace(np.log10(xmin),np.log10(np.max(degrees_friend50)+1), num=25, endpoint=True, base=10.0, dtype=int))
degrees_friend50=np.ndarray.flatten(np.array(degrees_friend50))  # To make a flat list(array) out of a list(array) of lists(arrays)

hist_friend50 = np.histogram(degrees_friend50, bins=mybins_friend50) # Histogram of the data
pdf_friend50 =hist_friend50[0]/np.sum(hist_friend50[0])#normalize histogram --for pdf


box_sizes=mybins_friend50[1:]-mybins_friend50[:-1]  # size of boxes
pdf_friend50 = pdf_friend50/box_sizes   # Divide pdf by its box size

#bins = Midpoints of distribution
mid_points_friend50=np.power(10, np.log10(mybins_friend50[:-1]) + (np.log10(mybins_friend50[1:]-1)-np.log10(mybins_friend50[:-1]))/2)




#Plot
fig,(ax1,ax2,ax3)=plt.subplots(nrows=1, ncols=3,figsize=(18, 6), sharex=False, sharey=False, squeeze= True)
# Set general font size
plt.rcParams['font.size'] = '20'
 #Plot
 


# Lambiotte teoretical plot
ax1.loglog(mid_points_friendq3,pdf_friendq3,color='blue',label = 'Numerical simulation', alpha=1,linestyle ='--',linewidth=2)
N = 1000 # Network size used to generate data
k = np.arange(N)
ax1.loglog(k,P3[N-1,],color='red',label = 'Theoretical calculations', alpha=1)

ax2.loglog(mid_points_friendq8,pdf_friendq8,color='blue',label = 'Numerical simulation', alpha=1, linestyle ='--',linewidth=2)
ax2.loglog(k,P8[N-1,],color='red',label = 'Theoretical calculations', alpha=1)

    
# Modified fof model plot
ax3.loglog(kn_and_Pkn[0],kn_and_Pkn[1],color='blue',label = 'Numerical simulation', alpha=1,linestyle ='--',linewidth=2)
#k=np.arange(3,len(ModifiedFOF_theoretical)+3)  # Theoretical Eqn is defined from k>=3
ax3.loglog(kt_and_Pkt[0], kt_and_Pkt[1],color='red',label = 'Theoretical simulation', alpha=1, linestyle ='-',linewidth=2)







    
# ax1.set(xlabel='Degree $k$', ylabel='$P_k$')
# ax2.set(xlabel='Degree $k$', ylabel='$P_K$')
# ax3.set(xlabel='Degree $k$', ylabel='$P_K$')



for ax in fig.get_axes():
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
 	    label.set_fontsize(20)
    ax.set(xlabel='Degree $k$', ylabel='Frequency')
    ax.label_outer()  # Set axis scales outer


### Setting the limits of the plot
ax1.axes.set_xlim([1,100])
ax1.axes.set_ylim([max(min(pdf_friendq3),min(P3[N-1,])),  max(max(pdf_friendq3),max(P3[N-1,]))])

ax2.axes.set_xlim([8,1000])
ax2.axes.set_ylim([max(min(pdf_friendq8),min(P8[N-1,])),   max(max(pdf_friendq8),max(P8[N-1,]))])

ax3.axes.set_xlim([10,  1000]) 
#ax3.axes.set_ylim([min(min(pdf_friend50),min(ModifiedFOF_theoretical)),   max(max(pdf_friend50),max(ModifiedFOF_theoretical))]) 
ax3.axes.set_ylim([0.00000001,0.1]) 


## Panels labels
ax1.text(0.6, 0.2, 'A', fontsize=20,fontweight='bold')
ax2.text(5, 0.005, 'B', fontsize=20,fontweight='bold')
ax3.text(6, 0.05, 'C', fontsize=20,fontweight='bold')
#Splt.savefig('LambiotteModifiedfof_loglog_plot2.pdf')


#plt.xlim([10,10000])  # Limiting x axis
#plt.ylim([min(min(pdf_friendq3),min(P3[N-1,]), min(pdf_friendq8),min(P8[N-1,]),min(pdf_friend50),min(ModifiedFOF_theoretical)),1])  # Limiting y axis
#plt.ylim([0.0000001,max_freq])  # Limiting x axis


# #determine axes and their limits
# axes= fig.get_axes() 
# ax_ylims = [(ax, ax.get_ylim()) for ax in axes]

# #find maximum y-limit spread
# max_freq = max([lmax-lmin for _, (lmin, lmax) in ax_ylims])


# # Lambiotte teoretical plot
# plt.loglog(mid_points_friendq3,pdf_friendq3,color='blue',label = 'Numerical simulation', alpha=1,linestyle ='--',linewidth=2)
# N = 1000 # Network size used to generate data
# k = np.arange(N)
# plt.loglog(k,P3[N-1,],color='red',label = 'Theoretical calculations', alpha=1)









