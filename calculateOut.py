import numpy as np

def calculateOut(x,p,q,s,c,m,n):
    a, b, y, w, u = 0, 0, np.zeros((m),dtype=float), np.ones((m),dtype=float), np.zeros((n,m),dtype=float)

    for j in range(m):
        for i in range(n):
            y[j] = ((p[i,j]*x[i]) + q[j] + y[j])
            u[i,j] = np.exp(-0.5*(((x[i]-c[i,j])/s[i,j])**2))
            w[j] = w[j] * u[i,j]
        a = a + (w[j]*y[j])
        b = b + w[j]
    ys = a / b
    return ys,y,w,b
