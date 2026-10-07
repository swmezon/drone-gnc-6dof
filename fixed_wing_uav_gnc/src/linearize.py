import numpy as np
from src.dynamics import derivative

def linearize(x0,u0,p,eps_x=1e-5,eps_u=1e-5):
    n=len(x0); m=len(u0)
    A=np.zeros((n,n)); B=np.zeros((n,m))
    for i in range(n):
        h=eps_x*max(1.0,abs(x0[i]))
        dx=np.zeros(n); dx[i]=h
        A[:,i]=(derivative(0,x0+dx,u0,p)-derivative(0,x0-dx,u0,p))/(2*h)
    for j in range(m):
        h=eps_u*max(1.0,abs(u0[j]))
        du=np.zeros(m); du[j]=h
        B[:,j]=(derivative(0,x0,u0+du,p)-derivative(0,x0,u0-du,p))/(2*h)
    return A,B

def modal_summary(A):
    eig=np.linalg.eigvals(A)
    rows=[]
    for lam in eig:
        wn=float(abs(lam))
        zeta=float(-lam.real/wn) if wn>1e-9 else float('nan')
        rows.append((lam.real,lam.imag,wn,zeta))
    return eig,rows
