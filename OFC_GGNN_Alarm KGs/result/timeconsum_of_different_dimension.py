import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']

x = [20,23,26,30,33,36,40]
x = np.array(x)
print('x is :\n',x)
num = [4.36,4.42,4.473,4.634,4.72,4.93,5.12]
y1 = np.array(num)

accuracy = [96.8,96.6,97.3,95.2,95.2,94.2,95.5]
y2 = np.array(accuracy)

print('y is :\n',y1)
#用3次多项式拟合
f1 = np.polyfit(x, y1, 3.5)
print('f1 is :\n',f1)

p1 = np.poly1d(f1)
print('p1 is :\n',p1)

#也可使用yvals=np.polyval(f1, x)
yvals = p1(x) #拟合y值
print('yvals is :\n',yvals)
#绘图


fig = plt.figure()

ax1 = fig.add_subplot(111)

ax1.scatter(x,y1,c = 'black', s=50, linewidths=1, marker='2',label='Original time_consum value' )
#ax1.plot(x, y1, 's',label='original time_consum value')
ax1.plot(x, yvals,'r',linewidth=1, label='Fitting curve')
ax1.set_ylabel('Average localization time(ms)',fontsize=12)
plt.xlabel('Dimension of hidden state',fontsize=12)
#plt.xticks(xticksig)#取消x轴刻度的显示
ax1.set_xticks(x.astype(int))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

ax2 = ax1.twinx()  # this is the important function
ax2.plot(x, y2,'b',linewidth=0.7, markersize=6, marker='x',label='Accuracy')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax2.set_ylabel('Accuracy(%)',fontsize=12)


plt.ylim(50, 100)
plt.xlim(19.8, 40.2)
#plt.xlabel('x')
#plt.ylabel('y')


fig.legend(loc=4,bbox_to_anchor=(1,0), bbox_transform=ax1.transAxes,fontsize=12) #指定legend的位置右下角
plt.title('The impact of hidden state dimensions on performance',fontsize=12)
plt.savefig('figure9.png',dpi=1200)
plt.show()