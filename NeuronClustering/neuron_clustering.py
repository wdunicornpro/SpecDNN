import torch
from torch import exp,zeros_like

def bernoulli(x,q,m):

    if x==0 or x==1:
        return x
    b = 1
    s = 1
    for i in range(m-q):
        b *= (m-i)/(i+1)*(1-x)
        s *= x
        s += b

    
    return s*(x**q)


def OSLOM_based(weights,kernel,t=0.99):

    wts = weights[kernel].abs()
    wts = wts.view(wts.shape[0],wts.shape[1],-1).mean(2)
    n,m = wts.shape
    print(m)
    r = exp(-m/2*wts.sum(0)/wts.sum().item()-0.5)
    Rq = sorted(list(range(m)),key=lambda x:r[x])
    Omega = [bernoulli(r[i].item(),i+1,m) for i in Rq]
    print(Omega[:10])
    # print(max(Omega))
    q = 1
    # while Omega[q] > t and q < m:
    #     q += 1
    # while Omega[q] < t and q < m:
    #     q += 1
    for i in range(m):
        if Omega[i] > t:
            q = i
            break
        elif i == m-1:
            q = m
    return Rq[:q]


def CCME_based(weights):

    return weights