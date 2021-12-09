# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
plt.rcParams['font.sans-serif'] = ['Times New Roman']

x = pd.read_excel('loss_of_different_T.xlsx', sheet_name='Sheet4', usecols=[0]).values
y = pd.read_excel('loss_of_different_T.xlsx', sheet_name='Sheet4', usecols=[4]).values
y2 = pd.read_excel('loss_of_different_T.xlsx', sheet_name='Sheet4', usecols=[7]).values
y3 = pd.read_excel('loss_of_different_T.xlsx', sheet_name='Sheet4', usecols=[8]).values
y4 = pd.read_excel('loss_of_different_T.xlsx', sheet_name='Sheet4', usecols=[9]).values
y5 = pd.read_excel('loss_of_different_T.xlsx', sheet_name='Sheet4', usecols=[10]).values
acc = pd.read_excel('loss_of_different_T.xlsx', sheet_name='Sheet4', usecols=[5]).values
k=0
z=np.zeros(600)

'''for i in range(55,99):
    #print(i)
    z[k]=y4[i]-0.1
    print(z[k])
    k=k+1'''



#print(z)
#print(y)

fig1 = plt.figure()
ax = fig1.add_subplot()
ax.plot(x,y2, color = '#fd411e',linewidth=0.7, label = 'T=5')
ax.plot(x,y3, color = '#a2cffe',linewidth=0.7, label = 'T=10')
ax.plot(x,y4, color = '#56fca2',linewidth=0.7, label = 'T=15')
ax.plot(x,y5, color = '#0c06f7',linewidth=0.7, label = 'T=20')
#ax.plot(x2,acc, color = 'black',linewidth=0.7, label = 'Loss5')
#ax.plot(x1,z, '-r',linewidth=0.7, label = 'Fitting_Loss')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('The loss curve of different training timesteps (AKG15)',fontsize=12)

fig1.legend(loc=4,bbox_to_anchor=(1,0.7), bbox_transform=ax.transAxes,fontsize=12)

ax.set_xlabel("Iteration",fontsize=12)
ax.set_ylabel("Loss",fontsize=12)

plt.savefig('figure7.png',dpi=1200)
plt.show()