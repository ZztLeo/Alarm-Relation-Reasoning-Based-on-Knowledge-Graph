# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
plt.rcParams['font.sans-serif'] = ['Times New Roman']

'''x1 = np.array(pd.read_excel('result.xlsx',usecols=[0]).values)
y1 = pd.read_excel('result.xlsx',usecols=[1]).values
y2 = pd.read_excel('result.xlsx',usecols=[2]).values
y3 = pd.read_excel('result.xlsx',usecols=[3]).values
y4 = pd.read_excel('result.xlsx',usecols=[4]).values
print(x1.size)
z1 = np.zeros((460,1))
z2 = np.zeros((460,1))
z3 = np.zeros((460,1))
z4 = np.zeros((460,1))
#print(x1)
#print(y1[3])
for i in range(x1.size):
    k1 = random.random()*0.2
    k2 = random.random()*0.1
    if (i>350):
        z1[i] = y1[i]+k2+0.2
    else:
        z1[i] = y1[i]+k1
    #print(k)
    if i < 4 or i>455:
        continue 
    z1[i] = (y1[i-4]+y1[i-3]+y1[i-2]+y1[i-1]+y1[i]+y1[i+1]+y1[i+2]+y1[i+3]+y1[i+4])/10
    #z2[i] = (y2[i-2]+y2[i-1]+y2[i]+y2[i+1]+y2[i+2])/5
    #z3[i] = (y3[i-2]+y3[i-1]+y3[i]+y3[i+1]+y3[i+2])/5
    #z4[i] = (y4[i-2]+y4[i-1]+y4[i]+y4[i+1]+y4[i+2])/5'''

T1 = [4.6,9.6,14.6, 19.6]
T2 = [5.4,10.4,15.4, 20.4]
accuracy1 = [96.38,97.25,87.13,80.34]
accuracy2 = [93.35,95.38,96.37,95.32]

fig = plt.figure()
ax3 = fig.add_subplot(111)
width=0.8
xmajorFormatter = FormatStrFormatter('%d')
xminorLocator  = MultipleLocator(5)
ax3.xaxis.set_major_locator(xminorLocator)
ax3.xaxis.set_major_formatter(xmajorFormatter)
ax3.bar(T1,accuracy1,width,edgecolor='black',color='#2c6fbb',label='AKG15')
#plt.xticks(fontsize=15)
#plt.yticks(fontsize=15)
ax3.bar(T2, accuracy2, width,edgecolor='black',color='#dc4d01',label='AKG40')
plt.xlim(xmin=3,xmax=22)
print(plt.ylim(ymin=50,ymax=100))
fig.legend(loc=4,bbox_to_anchor=(0.89,0.85), bbox_transform=ax3.transAxes,fontsize=10)
#fig.legend(loc=1, bbox_to_anchor=(0.29,1.4),fontsize=10)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax3.set_xlabel("Training timestep T",fontsize=12)
ax3.set_ylabel(r"Accuracy(%)",fontsize=12)
plt.title('The effect of different training timesteps',fontsize=12)

#ax = fig1.add_subplot(111)
#ax.plot(x1[4:455],z1[4:455], '-r', linewidth= 0.5, label = 'T=5')
#ax.plot(x1,y2, '-r', label = 'T=10')
#ax.plot(x1,y3, '-b', label = 'T=15')
#ax.plot(x1,y4, '-g', label = 'T=20')
plt.savefig('figure8.png',dpi=1200)
plt.show()