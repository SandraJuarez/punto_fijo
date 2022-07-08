from matplotlib import pyplot as plt
import numpy as np
import math as mt
from scipy.special import eval_legendre
from scipy import integrate
import pandas as pd
import csv
import os

SMALL_SIZE = 8
MEDIUM_SIZE = 12
BIGGER_SIZE = 14
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


dir_name='C:\\Users\52333\Documents\liquidos\programas_python\facil_mezclacte\gfig'
#plt.rcParams["savefig.directory"] = os.chdir(os.path.dirname(dir_name))

x = np.linspace(-1, 1, num=20001)
ang=np.arccos(x)
grad=ang*180/(mt.pi)
c_dict={}
g_dict={}
b_dict={}
n0=20
s=0.10

for nc in range(0,10):
  for i in range(0,6):
      name=round(s*100)
      name2=round(s,2)
      sstr=str(name)+str(n0)
      sstrc='cfig/'+str(name)+'_'+str(n0)
      sstrg='gfig/'+str(name)+'_'+str(n0)
      sstrb='bfig/'+str(name)+'_'+str(n0)
      if(i<=5):
          c_dict[i]=np.loadtxt(sstr+'gmcgb.txt',skiprows=1,usecols=1,delimiter=',  ')
          g_dict[i]=np.loadtxt(sstr+'gmcgb.txt',skiprows=1,usecols=2,delimiter=',  ')
          b_dict[i]=np.loadtxt(sstr+'gmcgb.txt',skiprows=1,usecols=3,delimiter=',  ')
      if(i==6):
          c_dict[6]=c_dict[5]
          g_dict[6]=g_dict[5]
          b_dict[6]=b_dict[5]
      c=c_dict[i]
      g=g_dict[i]
      b=b_dict[i]
      plt.plot(grad,g,'-',markersize=5,label=s)
      plt.title('n='+str(n0)+ ', s='+str(name2))
      plt.xlabel(r'$\theta$')
      plt.ylabel(r'$g(\theta)$')
      plt.savefig(sstrg+'g.png')
      plt.clf()

      plt.plot(grad,c,'-',markersize=5,label=s)
      plt.title('n='+str(n0)+ ', s='+str(name2))
      plt.xlabel(r'$\theta$')
      plt.ylabel(r'$c(\theta)$')
      plt.savefig(sstrc+'c.png')
      plt.clf()

      plt.plot(grad,b,'-',markersize=5,label=s)
      plt.title('n='+str(n0)+ ', s='+str(name2))
      plt.xlabel(r'$\theta$')
      plt.ylabel(r'$b(\theta)$')
      plt.savefig(sstrb+'b.png')
      plt.clf()
      n0=n0+20

  s=s+0.10
  n0=20
  g_dict.clear()
  c_dict.clear()
  b_dict.clear()

fig, ax = plt.subplots(1,3,sharex=True)


plt.clf()
g20=np.loadtxt('20120gmcgb.txt',skiprows=1,usecols=2,delimiter=',  ')
g40=np.loadtxt('40120gmcgb.txt',skiprows=1,usecols=2,delimiter=',  ')
g60=np.loadtxt('60120gmcgb.txt',skiprows=1,usecols=2,delimiter=',  ')
g80=np.loadtxt('80120gmcgb.txt',skiprows=1,usecols=2,delimiter=',  ')
g100=np.loadtxt('100120gmcgb.txt',skiprows=1,usecols=2,delimiter=',  ')

plt.plot(grad,g20,label='s=0.20')
#plt.plot(grad,g40)
plt.plot(grad,g60,label='s=0.60')
#plt.plot(grad,g80)
plt.plot(grad,g100,label='s=1.0')
plt.title(r'$g(\theta) \ N=120$')
plt.xlabel(r'$\theta$')
plt.ylabel(r'$g(\theta)$')
plt.legend()

axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels('')
axins.set_yticklabels('')

ax.indicate_inset_zoom(axins)

plt.show()

'''
#g120=np.loadtxt('120gant1.txt',skiprows=1,usecols=0)
g=np.loadtxt('20gant70.txt',skiprows=1,usecols=0)
gsig=np.loadtxt('7020gsig.txt',skiprows=1,usecols=1)
gant=np.loadtxt('7020gant.txt',skiprows=1,usecols=1)
g0=np.loadtxt('7020gmcgb.txt',skiprows=1,usecols=2,delimiter=',  ')
plt.plot(grad,g)
plt.plot(grad,gsig)
plt.plot(grad,gant)
plt.plot(grad,g0)
plt.show()
'''
