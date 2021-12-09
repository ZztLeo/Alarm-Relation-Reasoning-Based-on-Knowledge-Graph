# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
plt.rcParams['font.sans-serif'] = ['Times New Roman']

x1 = pd.read_excel('accuracy_loss.xlsx',usecols=[0]).values
y = pd.read_excel('accuracy_loss.xlsx',usecols=[1]).values
z = pd.read_excel('accuracy_loss.xlsx',usecols=[2]).values
x2 = pd.read_excel('accuracy_loss.xlsx',usecols=[4]).values
k = pd.read_excel('accuracy_loss.xlsx',usecols=[5]).values
node_number1 = [14.5,19.5,24.5]
node_number2 = [15.5,20.5,25.5]
locate_time = [4.56,7.32,8.8]
accuracy = [99.78,98.89,99.23]

fig1 = plt.figure()
ax = fig1.add_subplot(111)
ax.plot(x1,y, color = '0.8',linewidth=0.7, label = 'Loss')
ax.plot(x1,z, '-r',linewidth=0.7, label = 'Fitting_Loss')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title('The feasibility of the GGNN-based reasoning model',fontsize=12)

ax2 = ax.twinx()
ax2.plot(x2, k, 'black',linewidth=0.7, label = 'Accuracy')
fig1.legend(loc=1, bbox_to_anchor=(1,0.8), bbox_transform=ax.transAxes,fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.set_xlabel("Iteration",fontsize=12)
ax.set_ylabel(r"Loss",fontsize=12)
ax2.set_ylabel(r"Accuracy",fontsize=12)

plt.savefig('figure6.png',dpi=1200)

plt.show()

'''fig2 = plt.figure(DeprecationWarning)
ax3 = fig2.add_subplot(111)
width=1
xmajorFormatter = FormatStrFormatter('%d')
xminorLocator  = MultipleLocator(5)
ax3.xaxis.set_major_locator(xminorLocator)
ax3.xaxis.set_major_formatter(xmajorFormatter)
ax3.bar(node_number1,locate_time,width,color='dodgerblue',label='locate_time')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
ax4 = ax3.twinx()
ax4.bar(node_number2, accuracy,width,color='y',label='accuracy')
plt.xlim(xmin=10,xmax=30)
print(plt.ylim(ymin=95,ymax=100))
fig2.legend(loc=1, bbox_to_anchor=(0.29,1.4), bbox_transform=ax3.transAxes,fontsize=10)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
ax3.set_xlabel("Alarm Knowledge Graph Scale(node number)",fontsize=15)
ax3.set_ylabel(r"Locate time(ms)",fontsize=15)
ax4.set_ylabel(r"Accuracy(%)",fontsize=15)

plt.savefig('2.png')

plt.show()'''
#plt.close()
