import matplotlib.pyplot as plt
import numpy as np
import math as mt
from scipy.special import eval_legendre
from scipy import integrate
import pandas as pd

grad=np.loadtxt("grad.txt",usecols=0,delimiter=',')

#la función g se calculó con HNC
g=np.loadtxt("1520gmcgb.txt",usecols=2,delimiter=',',skiprows=1)
gsig=np.loadtxt("1520gsig.txt",usecols=1,delimiter=',',skiprows=1)
gant=np.loadtxt("1520gant.txt",usecols=1,delimiter=',',skiprows=1)
cf=np.loadtxt('1520gmcgb.txt',usecols=1,delimiter=',',skiprows=1)
n=20
l=0.15
#c2=np.loadtxt('100cm.txt',usecols=0)
#cargamos los parámetros
bt=np.loadtxt("parametros.csv",skiprows=1,usecols=0,delimiter=',')
a=np.loadtxt("parametros.csv",skiprows=1,usecols=1,delimiter=',')
k=np.loadtxt("parametros.csv",skiprows=1,usecols=2,delimiter=',')
b=np.loadtxt("parametros.csv",skiprows=1,usecols=3,delimiter=',')
al=np.loadtxt("parametros.csv",skiprows=1,usecols=4,delimiter=',')
#l=np.loadtxt("parametros.csv",skiprows=1,usecols=5,delimiter=',')
#n=np.loadtxt("parametros.csv",skiprows=1,usecols=6,delimiter=',')
dn=np.loadtxt("parametros.csv",skiprows=1,usecols=7,delimiter=',')


x = np.linspace(-1, 1, num=20001)

ang=np.arccos(x)
#s=np.sin(ang)
r=2*a*np.sin(ang/2)
np.place(r, r==0, [1e-9])

ar=4*mt.pi*a**2
print(n)

#calculamos la derivada de la tensión superficial con respecto a rho.
#utilizamos la fórmula de la compresibilidad
#tenemos que añadir un menos porque la variable ang (theta) está acomodada de pi a 0
#el 2pi que aparece multiplicando a las integrales sale de la integración en el ángulo phi
h=g-1.0
#a=1

#dtc=(1/(mt.pi*a**2*(integrate.simpson(h,x))+(4*mt.pi*a**2/n)))*(-16*mt.pi**2*a**4)/n*(-1/ar)
#dtc=-1.0/(n*integrate.simpson(h*s,ang)+1.0)
#dtc=1.0/(2*n*integrate.simpson(h,x)+1.0)
dtc=-0.5*n*integrate.simpson(cf,x,even='first')
#dtc=1-n*cm



##############################################################################################

n1=n-dn
n2=n+dn

#con ello obtenemos la derivada de la compresibilidad con respecto a rho
#de la derivada numérica tenemos la derivada respecto a n. Con regla de la cadena sale respecto a rho

#en la fórmula del virial hay una derivada del potencial. Calculamos esto numéricamente

u=np.loadtxt("potencial.txt",skiprows=1,usecols=0)

#du=b*np.exp(-r*(al+k))*(-np.exp(al*r)*(k*r+1)+al*r+k*r+1)*(1/r**2)
#du=b*np.exp(-al*r)*np.exp(-k*r)*((1-np.exp(al*r))*(k/r+1/r**2)+al/r)
#du=np.gradient(u,r)
du=-k*b*np.exp(-k*r)*(1-np.exp(-al*r))*(1.0/r)+b*np.exp(-k*r)*al*np.exp(-al*r)*(1.0/r)+b*np.exp(-k*r)*(1-np.exp(-al*r))*(-1.0/r**2)

#np.savetxt('du.txt', np.transpose([grad,du]),delimiter=',   ')
#tomamos la formula del virial calculada con n=60
#recordemos que 2pi sale de integrar en el angulo phi

#vir1=n1/ar-(n1**2/(8*mt.pi*a**2)**2)*(2*mt.pi*a**2*integrate.simpson(r*du*gant,x))
#du=np.flip(du)
f1=r*du*gant
f2=r*du*gsig
ddx=2.0/20001
vir1=-(n1**2/8)*integrate.simpson(f1,x)#*integrate.simpson(gant,ang)#*integrate.simpson(gant,x)
#después tomamos la fórmula del virial con n=60+dn
#vir2=n2/ar-(n2**2/(8*mt.pi*a**2)**2)*(2*mt.pi*a**2*integrate.simpson(r*du*gsig,x))
vir2=-(n2**2/8)*integrate.simpson(f2,x)#*integrate.simpson(gsig,ang)#*integrate.simpson(gsig,x)
print(vir2)
print(vir1)
print(vir2-vir1)
#y utilizamos la fórmula de derivada numérica
der=(vir2-vir1)/(2*dn)
print(der)

dtv=der

#calculamos la tensión superficial
tau=n-(n**2/2)*integrate.simpson(r*du*g,x)


df=pd.DataFrame([{'n':n,'b':b,'l':l,'camino compresibilidad':dtc,'camino virial':dtv,'Tensión superficial':tau}],columns=['n','b','l','camino compresibilidad','camino virial','Tensión superficial'],dtype=float)
print(df)
df.to_csv('resultados2.csv',mode='a',header=False,index=False)
