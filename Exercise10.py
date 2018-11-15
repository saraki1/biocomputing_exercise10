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

###Custom likelihood function
def nlike(p,obs):
    B0=p[0]
    