import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from numpy.random import rand
from pylab import figure

"""
data = pd.read_csv("Z:\\comma_group1\\User\\fcp18asy\\PhD Work\\Papers\\SA_for_CPP-Aaron\\Figures\\Exp_Proposals\\Data.csv", header=[0, 1], index_col=0)
m1 = data.values
data2 = pd.read_csv("Z:\\comma_group1\\User\\fcp18asy\\PhD Work\\Papers\\SA_for_CPP-Aaron\\Figures\\Exp_Proposals\\Data2.csv", header=[0, 1], index_col=0)
m2 = data2.values
data3 = pd.read_csv("Z:\\comma_group1\\User\\fcp18asy\\PhD Work\\Papers\\SA_for_CPP-Aaron\\Figures\\Exp_Proposals\\Data3.csv", header=[0, 1], index_col=0)
m3 = data3.values
full_data = pd.read_csv("Z:\\comma_group1\\User\\fcp18asy\\PhD Work\\Papers\\SA_for_CPP-Aaron\\Figures\\Exp_Proposals\\Full_Data.csv", header=[0, 1], index_col=0)
m = full_data.values
fig = figure()
ax = fig.add_subplot(111, projection = "3d")

i = 0
print(m[i,:])
ax.scatter(m[i, 0], m[i, 1], m[i, 2], marker='+', color='b', label="Exps to Identify Critical Conditions")
ax.text(m[i, 0], m[i, 1], m[i, 2],  '%s' % (str(i+1)), size=20, zorder=1, color='k')

i = 1
print(m[i,:])
ax.scatter(m[i, 0], m[i, 1], m[i, 2], marker='+', color='b')
ax.text(m[i, 0], m[i, 1], m[i, 2],  '%s' % (str(i+1)), size=20, zorder=1, color='k')

i = 2
print(m[i,:])
ax.scatter(m[i, 0], m[i, 1], m[i, 2], marker='*', color='r', label="Critical Conditions Exps 3, 7 and 8")
ax.text(m[i, 0], m[i, 1], m[i, 2],  '%s' % ("C"), size=20, zorder=1, color='k')
# r'$\times 3$'
i = 3
print(m[i,:])
ax.scatter(m[i, 0], m[i, 1], m[i, 2], marker='+', color='b')
ax.text(m[i, 0], m[i, 1], m[i, 2],  '%s' % (str(i+1)), size=20, zorder=1, color='k')

i = 4
print(m[i,:])
ax.scatter(m[i, 0], m[i, 1], m[i, 2], marker='+', color='b')
ax.text(m[i, 0], m[i, 1], m[i, 2],  '%s' % (str(i+1)), size=20, zorder=1, color='k')

i = 5
print(m[i,:])
ax.scatter(m[i, 0], m[i, 1], m[i, 2], marker='+', color='b')
ax.text(m[i, 0], m[i, 1], m[i, 2],  '%s' % (str(i+1)), size=20, zorder=1, color='k')

i = 8
print(m[i,:])
ax.scatter(m[i, 0], m[i, 1], m[i, 2], marker='x', color='g', label="Exps Investigate Kinetics at Critical Conditions")
ax.text(m[i, 0], m[i, 1], m[i, 2],  '%s' % (str(i+1)), size=20, zorder=1, color='k')

i = 9
print(m[i,:])
ax.scatter(m[i, 0], m[i, 1], m[i, 2], marker='x', color='g')
ax.text(m[i, 0], m[i, 1], m[i, 2],  '%s' % (str(i+1)), size=20, zorder=1, color='k')

i = 10
print(m[i,:])
ax.scatter(m[i, 0], m[i, 1], m[i, 2], marker='x', color='g')
ax.text(m[i, 0], m[i, 1], m[i, 2],  '%s' % (str(i+1)), size=20, zorder=1, color='k')
"""
full_data = pd.read_csv("Z:\\comma_group1\\User\\fcp18asy\\PhD Work\\Papers\\SA_for_CPP-Aaron\\Figures\\Exp_Proposals\\Pre_Data.csv", header=[0, 1], index_col=0)
m = full_data.values
fig = figure()
ax = fig.add_subplot(111, projection = "3d")

i = 0
print(m[i,:])
ax.scatter(m[i, 0], m[i, 1], m[i, 2], marker='+', color='b', label="High and Low Exps "r'$\times 2$')
ax.text(m[i, 0], m[i, 1], m[i, 2],  '%s' % (str(i+1)), size=20, zorder=1, color='k')

i = 1
print(m[i,:])
ax.scatter(m[i, 0], m[i, 1], m[i, 2], marker='+', color='b')
ax.text(m[i, 0], m[i, 1], m[i, 2],  '%s' % (str(i+1)), size=20, zorder=1, color='k')

i = 2
print(m[i,:])
ax.scatter(m[i, 0], m[i, 1], m[i, 2], marker='+', color='b')
ax.text(m[i, 0], m[i, 1], m[i, 2],  '%s' % (str(i+1)), size=20, zorder=1, color='k')

i = 3
print(m[i,:])
ax.scatter(m[i, 0], m[i, 1], m[i, 2], marker='+', color='b')
ax.text(m[i, 0], m[i, 1], m[i, 2],  '%s' % (str(i+1)), size=20, zorder=1, color='k')

i = 4
print(m[i,:])
ax.scatter(m[i, 0], m[i, 1], m[i, 2], marker='+', color='b')
ax.text(m[i, 0], m[i, 1], m[i, 2],  '%s' % (str(i+1)), size=20, zorder=1, color='k')

i = 5
print(m[i,:])
ax.scatter(m[i, 0], m[i, 1], m[i, 2], marker='+', color='b')
ax.text(m[i, 0], m[i, 1], m[i, 2],  '%s' % (str(i+1)), size=20, zorder=1, color='k')

i = 6
print(m[i,:])
ax.scatter(m[i, 0], m[i, 1], m[i, 2], marker='+', color='b')
ax.text(m[i, 0], m[i, 1], m[i, 2],  '%s' % (str(i+1)), size=20, zorder=1, color='k')

i = 7
print(m[i,:])
ax.scatter(m[i, 0], m[i, 1], m[i, 2], marker='+', color='b')
ax.text(m[i, 0], m[i, 1], m[i, 2],  '%s' % (str(i+1)), size=20, zorder=1, color='k')

i = 8
print(m[i,:])
ax.scatter(m[i, 0], m[i, 1], m[i, 2], marker='x', color='g', label="Midpoint Exps "r'$\times 3$')
ax.text(m[i, 0], m[i, 1], m[i, 2],  '%s' % ("M"), size=20, zorder=1, color='k')

"""
i = 9
print(m[i,:])
ax.scatter(m[i, 0], m[i, 1], m[i, 2], marker='*', color='r')
ax.text(m[i, 0], m[i, 1], m[i, 2],  '%s' % ("M"), size=20, zorder=1, color='k')

i = 10
print(m[i,:])
ax.scatter(m[i, 0], m[i, 1], m[i, 2], marker='*', color='r')
ax.text(m[i, 0], m[i, 1], m[i, 2],  '%s' % ("M"), size=20, zorder=1, color='k')
"""

# cbar = plt.colorbar(pnt3d)
# cbar.set_label("Liquid Spray Rate, " r'$\dot{V}$, ' r'$\left[ g min^{-1} \right]$')
ax.set_xlabel("Impeller Frequency, " r'$n_{imp}$, ' r'$\left[ min^{-1} \right]$')
ax.set_ylabel("Liquid to Solid Ratio, " r'$L/S$, ' r'$\left[ kg \ kg^{-1} \right]$')
ax.set_zlabel("Kneading Time, " r'$t_{kn}$, ' r'$\left[ s \right]$')
# plt.legend(loc='best');
plt.legend(bbox_to_anchor=(0, 0.95), loc='lower left',
           borderaxespad=0.)
# plt.show()
plt.savefig("Z:\\comma_group1\\User\\fcp18asy\\PhD Work\\Papers\\SA_for_CPP-Aaron\\Figures\\Exp_Proposals\\pre-graph_raw.png")


"""
x_a = [370, 370, 370, 370, 310, 310]
y_a = [0.13, 0.15, 0.17, 0.19, 0.15, 0.19]
z_a = [300, 300, 300, 300, 300, 300]
c_a = [71, 71, 71, 71, 71, 71]
n_a = [1, 2, 3, 4, 5, 6]

x_crit = [370, 370]
y_crit = [0.17, 0.17]
z_crit = [300, 300]
c_crit = [71, 71]
n_crit = [7, 8]

x_b = [370, 370, 370]
y_b = [0.17, 0.17, 0.17]
z_b = [0, 60, 180]
c_b = [71, 71, 71]
n_b = [9, 10, 11]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x_a, y_a, z_a, c=c_a, marker='o')
ax.scatter(x_b, y_b, z_b, c=c_b, marker='X')
ax.scatter(x_crit, y_crit, z_crit, c=c_crit, marker='*')

# cbar=plt.colorbar(graph)
# cbar.set_label("Liquid Spray Rate, " r'$\dot{V}$, ' r'$\left[ g min^{-1} \right]$')

ax.set_xlabel("Impeller Frequency, " r'$n_{imp}$, ' r'$\left[ min^{-1} \right]$')
ax.set_ylabel("Liquid to Solid Ratio, " r'$L/S$, ' r'$\left[ kg \ kg^{-1} \right]$')
ax.set_zlabel("Kneading Time, " r'$t_{kn}$, ' r'$\left[ s \right]$')
plt.show()
"""
"""
for i, txt in enumerate(exp):
    ax.annotate(txt, (n_imp[i], L_S[i]))
"""



"""
m=rand(3,3) # m is an array of (x,y,z) coordinate triplets

fig = figure()
ax = Axes3D(fig)


for i in range(len(m)): #plot each point + it's index as text above
 ax.scatter(m[i,0],m[i,1],m[i,2],color='b')
 ax.text(m[i,0],m[i,1],m[i,2],  '%s' % (str(i)), size=20, zorder=1,
 color='k')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
pyplot.show()
"""