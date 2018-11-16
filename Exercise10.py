#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 18:02:27 2018

@author: stephaniearaki
"""

########EXERCISE 10############

######Question 1: Maximum Likelihood 

###Import packages
import numpy
import pandas
from scipy.optimize import minimize
from scipy.stats import norm
from plotnine import *

###Import data
data=pandas.read_csv('data.txt', sep=',',header=0)

###Visualize observations
ggplot(data,aes(x='x',y='y'))+geom_point()+theme_classic()

###Custom likelihood function: a+bx
def nllike(p,obs):
    B0=p[0]
    B1=p[1]
    sigma=p[2]
    
    expected=B0+B1*obs.x
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll
    
###Estimate parameters by minimizing neg log likelihood
initialGuess=numpy.array([1,1,1])
fit=minimize(nllike,initialGuess,method='Nelder-Mead',options={'disp':True},args=data)

print fit.x

###Custom likelihood function: a+bx+cx**2
def nllike2(p,obs):
    B0=p[0]
    B1=p[1]
    B2=p[2]
    sigma=p[3]
    
    expected=B0+B1*obs.x+B2*((obs.x)**2)
    nll=-1*norm(expected,sigma).logpdf(obs.y).sum()
    return nll

###Estimate parameters
initialGuess2=numpy.array([1,1,1,1])
fit2=minimize(nllike2,initialGuess2,method='Nelder-Mead',options={'disp':True},args=data)

print fit2.x
###Likelihood ratio test
from scipy import stats

teststat=2*(fit.fun-fit2.fun)
df=len(fit2.x)-len(fit.x)
pval=1-stats.chi2.cdf(teststat,df)

print pval

#linear (a+bx) is more appropriate because adding another term in the equation
    #did not make the model more accurate (pval higher, closer to 1)


######Question 2: Competition

###Import packages
import pandas
import scipy
import scipy.integrate as spint
from plotnine import *

###Write custom function
def ddSim(y,t0,R1,a11,a12,R2,a22,a21): 
    N1=y[0] 
    N2=y[1]
    dN1dt=R1*(1-N1*a11-N2*a12)*N1
    dN2dt=R2*(1-N2*a22-N1*a21)*N2
    return [dN1dt,dN2dt]

###Define parameters 1
N0=[0.1,0.1]
times=range(0,100)
params=(0.5,0.6,0.5,0.5,0.7,0.5) #r values between 0 and 1, alpha < 1
###Simulate model 1
modelSim1=spint.odeint(func=ddSim,y0=N0,t=times,args=params)
###Put model output in dataframe 1
modelOutput1=pandas.DataFrame({"t":times,"N1":modelSim1[:,0],"N2":modelSim1[:,1]})
###Plot simulation output 1
a=ggplot(modelOutput1,aes(x="t",y="N1"))+geom_line()+theme_classic()
a+geom_line(modelOutput1,aes(x="t", y="N2"),color="red")

###Define parameters 2
N0=[0.1,0.1]
times=range(0,100)
params2=(1,0.7,0.6,0.5,0.9,0.7)
###Simulate model 2
modelSim2=spint.odeint(func=ddSim,y0=N0,t=times,args=params2)
###Put model output in dataframe 2
modelOutput2=pandas.DataFrame({"t":times,"N1":modelSim2[:,0],"N2":modelSim2[:,1]})
###Plot simulation output 2
b=ggplot(modelOutput2,aes(x="t",y="N1"))+geom_line()+theme_classic()
b+geom_line(modelOutput2,aes(x="t", y="N2"),color="red")

###Define parameters 3
N0=[0.1,0.1]
times=range(0,100)
params3=(0.5,0.9,0.5,1,0.5,0.3) 
###Simulate model 3
modelSim3=spint.odeint(func=ddSim,y0=N0,t=times,args=params3)
###Put model output in dataframe 2
modelOutput3=pandas.DataFrame({"t":times,"N1":modelSim3[:,0],"N2":modelSim3[:,1]})
###Plot simulation output 3
c=ggplot(modelOutput3,aes(x="t",y="N1"))+geom_line()+theme_classic()
c+geom_line(modelOutput3,aes(x="t", y="N2"),color="red")
