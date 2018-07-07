
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math as m
from numba import jit
import timeit as tt
from scipy import signal


# In[2]:


# Double well
def u2(a,x):
    return a*(x+1)**2*(x-1)**2


# In[3]:


@jit #Following Todd's thesis for UNDERDAMPED Langevin
def itT(n, t0,tf, dt, gam, kBT,a , b, f, w0, wf, dw): #n trags

    # make all n trags and dW ahead of time
    tLength = m.ceil((tf-t0)/dt)
    wLength = m.ceil((wf-w0)/dw) +1
    tTab = np.arange(t0, tf, dt)
    wTab = np.arange(w0, wf+dw, dw)
    trags = np.empty([wLength, n, tLength])
    vt = np.empty([wLength, n, tLength])
    dWn = np.random.normal(0, (kBT*(1-m.exp(-gam*dt)))**(1/2), [wLength, n,tLength-1,2])
    bcounts, em, ep = np.zeros([wLength,n]), -(1-(1-b/a)**(1/2))**(1/2), (1-(1-b/a)**(1/2))**(1/2)
    rnc = ((2/(gam*dt))*m.tanh(gam*dt/2))**(1/2)
    for k in np.arange(wLength):
        for i in np.arange(n): # Outer loop over n/ pB/M/E = pBegin/M/E
            trags[k][i][0], xLast, pB, crossemp = 1.0, 1.0, 0.0, [False, False]
            vt[k][i][0] = pB
            for j in np.arange(tLength-1):   # Nested loop over length of individual trag x_j+1 = x_j + F(x_j,t_j) + dW_j
                pM = pB*m.exp(-gam*dt/2) + dWn[k][i][j][0] + gam**-1*(4*a*trags[k][i][j]*(1-trags[k][i][j]**2) -f*np.sin((w0+k*dw)*j*dt))*dt*rnc/2
                xNext = trags[k][i][j] + pM*rnc*dt
                trags[k][i][j+1] = xNext
                pB = (pM + gam**-1*(4*a*trags[k][i][j+1]*(1-trags[k][i][j+1]**2) -f*np.sin((w0+k*dw)*(j+1)*dt))*dt*rnc/2)*m.exp(-gam*dt/2) + dWn[k][i][j][1]
                vt[k][i][j+1] = pB
                #check to see if crossed:
                if xNext < xLast: # if we're moving to the left
                    if crossemp == [False, False]: # we havent crossed em or ep yet so only have to check ep
                        if xLast > ep:
                            if xNext < ep:
                                crossemp[1] = True
                            if xNext < em:
                                crossemp = [True, True] #if we passed through whole neighborhood
                    elif crossemp == [False, True]: #passed ep from left but not em
                        if xNext < em:
                            crossemp = [True,True]
                    elif crossemp == [True, False]:
                        if xNext < em:
                            crossemp = [False, False]

                else: #moving to the right
                    if crossemp == [False, False]:
                        if xLast < em:
                            if xNext > em:
                                crossemp[0] = True
                            if xNext > ep:
                                crossemp = [True,True]
                    elif crossemp == [True, False]:
                        if xNext > ep:
                            crossemp = [True,True]
                    elif crossemp == [False, True]:
                        if xNext > ep:
                            crossemp = [False, False]


        #if we've passed through the whole B-epsilon, reset and add to bcounts
                if crossemp == [True, True]:
                    bcounts[k][i], crossemp = bcounts[k][i] + 1, [False, False]
                xLast  = xNext

    return trags, tTab, wTab, bcounts, vt


# In[4]:


def pxnn(itab, res, n, bins): #construct and plot the average hist over n trags given w (res) in itab
    xL = bins+1
    htab = np.empty([n, xL])

    for i in np.arange(n):
        ht =  np.histogram(itab[0][res][i], bins , density =1, )

        htN = np.append(ht[0], ht[0][-1])

        for j in np.arange(len(htN)):

            #first make nested list of histogram data vs and then average at each x to
            htab[i][j] = htN[j]

    meanh = np.mean(htab, axis = 0)
    stdh = np.std(htab, axis = 0)
    xa=np.arange(-1.5,1.5,0.01)
    plt.errorbar(ht[1], meanh, yerr=stdh)
    plt.semilogy(xa,np.exp(np.add(-u2(4,xa),0.01)))
    plt.ylabel('$\ln p(x)$')
    plt.xlabel('$x$')
    plt.show()
    return ht[1], meanh


# In[33]:


rw1 = itT(100,0.0,1000.0, 0.01 , 1.0, 1.0, 4.0, 1.5, 2.2, 4.8,4.8, 0.1)


# In[19]:


px1=pxnn(rw1, 0, 100,100)


# In[5]:


#this works REALLY FAST!!!
@jit
def samptfast(bigtrag,bigttab, n,c,f,w):

    finalstates = bigtrag[n-1:n-1+n*(c-1)+1]
    worktab = np.zeros(n*(c-1)+1)
    for i in np.arange(n*(c-1)+1):
        for j in np.arange(n):
            worktab[i] = worktab[i]+ f*(m.sin(w*bigttab[j+1+i])- m.sin(w*bigttab[j+i]))*bigtrag[i+j]



    return worktab, finalstates


# In[5]:


samptfast([1,2,3,4,5,6,7,8,9], [1,2,3,4,5,6,7,8,9], 3,3,1,1)


# In[6]:


@jit
def wcumul( bs, bt, n, c, f, w ):

    #start by calculating a single long trag

    #sample on a log scale the smal trag length n
    bins = np.arange(-2.0, 2.0, 0.1)
    wbin = np.zeros((len(bins),3))
    wc = np.empty([len(bins),2])
    stf = samptfast(bs,bt, n,c,f,w)
    xindex=np.digitize(stf[1], bins)
    xdata = np.empty(len(xindex))
    for f in np.arange(len(xindex)):
        xdata[f] = bins[f]

    for q in np.arange(len(xindex)):
        wbin[xindex[q]][0] = wbin[xindex[q]][0]+stf[0][q]
        wbin[xindex[q]][1] = wbin[xindex[q]][0]+stf[0][q]**2
        wbin[xindex[q]][2] = wbin[xindex[q]][2] + 1

    for h in np.arange(len(bins)):

        if wbin[h][2]>0:
            wc[h][0]= wbin[h][0]/(wbin[h][2])
            wc[h][1] = wbin[h][1]/(wbin[h][2]) - wc[h][0]**2




    return wc


# In[7]:


run0 = itT(1, 0.0, 12000, 0.01, 1.0, 1.0,4.0 , 1.5, 2.2, 4.8,4.8, 0.1)
bs = run0[0][0][0][200000:]
bt = run0[1][200000:]


# In[ ]:


wc0 = wcumul(bs, bt, 1000, 1000, 2.2, 4.8)


# In[9]:


bins = np.arange(-2, 2, 0.1)


# In[30]:


wc0[:,1],wc0


# In[31]:


plt.plot(bins,-u2(4.0, bins)+wc0[:,0])
plt.plot(px1[0], np.log(px1[1])-np.max(np.log(px1[1]))+9)
plt.plot(bins,-u2(4.0, bins)-np.max(-u2(4.0, bins))+9)

plt.axis([-1.5, 1.5, 4, 10])
