# -*- coding: utf-8 -*-
"""
@author: Joseph Melville
Code for Small Satellite Conference 2022 article "Methods for Data-centric Small Satellite Anomaly Detection and Fault Prediction"
"""



### IMPORT LIBRARIES
import numpy as np
import functions as sf



### SETUP
t_new = sf.time_arr(date_start='2020-01-01 00:00:00', date_end='2021-01-01 00:00:00', dt=60) #seconds
t_min = t_new[0]
t_max = t_new[-1]
num_faults = 100
num_outliers = 1000
avg_dist = (t_max-t_min)/num_faults #average distance between faults
t_faults = np.random.randint(t_min, t_max, num_faults)



### TESTS

#Randomly set outlier times should not give high scores
t_outliers = np.random.randint(t_min, t_max, num_faults)
pred = sf.predictability_metrics(t_faults, t_outliers, t_new, if_plot=True)

#Outlier times exactly equal to the fault times should be perfect
t_outliers = t_faults
pred = sf.predictability_metrics(t_faults, t_outliers, t_new, if_plot=True)

#Outlier times set to one hour before only half of the fault times, outlier-to-fault should still be great
t_outliers = t_faults[:50] 
pred = sf.predictability_metrics(t_faults, t_outliers, t_new, if_plot=True)

#Outlier times set to one hour before only half of the fault times, outlier-to-fault should still be great
t_outliers = t_faults
pred = sf.predictability_metrics(t_faults[:50], t_outliers, t_new, if_plot=True)

#Outlier times set to one hour before each fault should be perfect, with a non-zero mean
t_outliers = t_faults - 60*60
pred = sf.predictability_metrics(t_faults, t_outliers, t_new, if_plot=True)

#Add some variabliity the the placement of the oultiers (should lower precision)
t_outliers = []
for f in t_faults:
    tmp = np.random.normal(loc=-60*60, scale=10, size=10)
    t_outliers.append(f + tmp)
t_outliers = np.hstack(t_outliers)
pred = sf.predictability_metrics(t_faults, t_outliers, t_new, if_plot=True)

#Investigate how the outlier distance in proportion to average distance before fault affects the estimate of the outlier distance
tmp1 = avg_dist/10 #1, 2, 10, 100
t_outliers = []
for f in t_faults:
    tmp = np.random.normal(loc=-tmp1, scale=50, size=10)
    t_outliers.append(f + tmp)
t_outliers = np.hstack(t_outliers)
pred = sf.predictability_metrics(t_faults, t_outliers, t_new, if_plot=True)
print('Mean time to fault is %3.2f %% actual'%(100*pred[0]/(tmp1/60/60))) #57.71, 75.18, 95.16, 99.35