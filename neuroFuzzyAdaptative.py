from scipy.io import loadmat
import numpy as np
from calculateOut import calculateOut
import matplotlib.pyplot as plt 

xt = loadmat('dinamico/xt2.mat')
xv = loadmat('dinamico/xv2.mat')
ydt = loadmat('dinamico/ydt2.mat')
ydv = loadmat('dinamico/ydv2.mat')

xt = xt['xt2']
xv = xv['xv2']
ydt = ydt['ydt2']
ydv = ydv['ydv2']

m=5
alfa=0.01
n=len(xt[0])
nEp=8
npt= len(ydt[0])
npv= len(ydv)

xmin = np.amin(xt,axis=0)
xmax = np.amax(xt,axis=0)
delta=(xmax-xmin)/(m-1)

ys = np.zeros((npt),dtype=float)

c,s,p,q = np.zeros((n,m),dtype=float),np.zeros((n,m),dtype=float),np.zeros((n,m),dtype=float),np.zeros((m),dtype=float)

for j in range(m):
    for i in range(n):
        c[i,j] = (xmin[i] + (j)*delta[i])
        s[i,j] = (0.5*delta[i]*np.sqrt(1/(2*np.log(2))))
        p[i,j] = np.random.rand()
    q[j] = np.random.rand()

for l in range(nEp):
    for k in range(npt):
        ys[k],y,w,b = calculateOut(xt[k],p,q,s,c,m,n)
        for j in range(m):
            dysdwj = (y[j] -  ys[k]) / b
            dysdyj = w[j] / b
            dedys = ys[k] - ydt[0,k]
            dydqj = 1
            for i in range(n):
                dwdcij = w[j]*((xt[k,i]-c[i,j])/(s[i,j]**2))
                dwdsij = w[j]*(((xt[k,i]-c[i,j])**2)/(s[i,j]**3))
                dydpij = xt[k,i]
                dedcij = dedys*dysdwj*dwdcij
                dedsij = dedys*dysdwj*dwdsij
                dedpij = dedys*dysdyj*dydpij
                dedqij = dedys*dysdyj*dydqj
            
                c[i,j] = c[i,j] - (alfa*dedcij)
                s[i,j] = s[i,j] - (alfa*dedsij)
                p[i,j] = p[i,j] - (alfa*dedpij)
                q[j] = q[j] - (alfa*dedqij)


ysaida = np.zeros((npv),dtype=float)

for i in range(npv):
    ysaida[i],y,w,b = calculateOut(xv[i],p,q,s,c,m,n)

plt.plot(ysaida,label = 'Modelo Treinado')
plt.plot(ydv,label = 'Modelo de Validação')

plt.legend()
plt.show() 
