# -*- coding: utf-8 -*-
"""
@author: Joseph Melville
Code for Small Satellite Conference 2022 article "Methods for Data-centric Small Satellite Anomaly Detection and Fault Prediction"
"""



### IMPORT LIBRARIES
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
import datetime
import time
import matplotlib.pyplot as plt



### FUNCTIONS
# def form_to_unix(formatted_time):
#     #convert from formatted timestamps to unix time (in seconds)
#     #e.g. utc_time = list_code(encoded_formatted_utc_time[:].tolist())
#     return np.array([time.mktime(datetime.datetime.strptime(s.split('.')[0].replace('T', ' ').replace('Z', ''), '%Y-%m-%d %H:%M:%S').timetuple()) for s in list_code(formatted_time[:].tolist())])



# def time_arr(date_start, date_end, dt=60):
#     #Returns a numpy array of unix time stamp in the range [date_start, date_end] (including date_end if divisible by dt)
#     #e.g. date_start='2020-02-01 00:00:00' (%YYYY-%mm-%dd %HH:%MM:%SS)
#     #dt is in seconds, return values are in seconds
#     time_start = form_to_unix(np.array([date_start.encode()]))[0]
#     time_end = form_to_unix(np.array([date_end.encode()]))[0]
#     return np.arange(time_start, time_end+dt/2, dt)



def form_to_unix(formatted_time):
    #Converts from formatted timestamps to unix time (in seconds)
    return time.mktime(datetime.datetime.strptime(formatted_time, '%Y-%m-%d %H:%M:%S').timetuple())



def time_arr(date_start, date_end, dt=60):
    #Returns a numpy array of unix time stamp in the range [date_start, date_end] (including date_end if divisible by dt)
    #e.g. date_start='2020-02-01 00:00:00' (%YYYY-%mm-%dd %HH:%MM:%SS)
    #dt is in seconds, return values are in seconds
    time_start = form_to_unix(date_start)
    time_end = form_to_unix(date_end)
    return np.arange(time_start, time_end+dt/2, dt)



def my_sort(arr_list, sort_by_indx=0):
    #Sorts a list of arrays together
    ixs = np.argsort(arr_list[0])
    arr_list_sorted = []
    for a in arr_list: arr_list_sorted.append(a[ixs])
    return arr_list_sorted



def find_indices_in_range(arr, low_value, high_value):
    #Returns the indices in the array 'arr' that are between and including the 'low_value' and 'high_value'
    #'arr' is assumped monotonically increasing
    indx = []
    for i,e in enumerate(arr):
        if (e>=low_value) and (e<high_value): indx.append(i)
        if e>high_value: break 
    return np.array(indx, dtype='int')



def resample_time(t_new,  t, arr, agg_func=np.mean):
    #Aggregates all the values in arr with timestamps in time regions defined by t_new
    dt = t_new[1]-t_new[0]
    [t_sorted, a_sorted] = my_sort([t,arr]) #Sort to help with searching
    
    #Trim arrays to the only feasible range
    first_low_value = t_new[0]-dt/2
    last_high_value = t_new[-1]+dt/2
    keep_i = np.logical_and(t_sorted<=last_high_value, t_sorted>=first_low_value)
    t_sorted = t_sorted[keep_i]
    a_sorted = a_sorted[keep_i]

    arr_new = []
    indx_curr = 0
    for e in t_new: 
        low_value = e-dt/2
        high_value = e+dt/2
        indx = find_indices_in_range(t_sorted[indx_curr:], low_value, high_value)
        if len(indx)==0: arr_new.append(np.nan)
        else: arr_new.append(agg_func(a_sorted[indx+indx_curr]))
        if len(indx)>0: indx_curr+=indx[-1]+1
        
    return np.array(arr_new)



def blip_outliers(a, win_size=600, stride=50):
    #Finds oultiers based on how few values are found in a time region
    #Returns timestamps for this outliers
    is_outlier = np.zeros(a.shape)
    for i in range(win_size, len(a), stride):
        tmp = a[i-win_size:i]
        num_not_nan = np.sum(np.isnan(tmp)==False)
        if np.logical_and(num_not_nan<=5, num_not_nan!=0):
            is_outlier[i-win_size:i] = 1
    
    tmp = a[len(a)-win_size:len(a)]
    num_not_nan = np.sum(np.isnan(tmp)==False)
    if np.logical_and(num_not_nan<=5, num_not_nan!=0):
        is_outlier[len(a)-win_size:len(a)] = 1
    
    return is_outlier



def get_outliers(v, t, t_new):
    #"v" is a numpy array of values
    #"t" is a numpy array of timsetamps for thos values
    #"t_new" is the common timeline for resampling
    #Returns timestamps for outliers find using four different methods
    if len(v)<3: 
        print('Cannot find outliers for less than 3 entries')
        return [np.array([]), ]*4
    
    #Reduce dimensions, if not already 2 or less
    pca = PCA(n_components=2) 
    if v.shape[1]>2: v_PCA = pca.fit_transform(v)
    else: v_PCA = v
   
    #Find Blip oultiers
    v_rs = resample_time(t_new,  t, v[:,0], agg_func=np.mean)
    outliers_blip = blip_outliers(v_rs) 
    outliers_blip = t_new[outliers_blip==1]
   
    #Find IF outliers
    clf = IsolationForest(random_state=0)
    outliers_IF = clf.fit_predict(v)==-1
    outliers_IF = t[outliers_IF]
    
    #Find IF outliers, with PCA calues
    clf = IsolationForest(random_state=0)
    outliers_IF_PCA = clf.fit_predict(v_PCA)==-1
    outliers_IF_PCA = t[outliers_IF_PCA]
    
    #Find IF outliers, with PCA calues
    clf = LocalOutlierFactor(n_neighbors=5)
    clf.fit(v_PCA)
    lof = clf.negative_outlier_factor_
    clf = IsolationForest(random_state=0)
    outliers_LOF = clf.fit_predict(lof[:,None])==-1
    outliers_LOF = t[outliers_LOF]
    
    return [outliers_blip, outliers_IF, outliers_IF_PCA, outliers_LOF]



def dist_to_next_in_range(t, arr, low_value=1, high_value=1):
    #Finds the distance in time between each value in 'arr' and the next value in 'arr' that falls betweeeen 'low_value' and 'high_value'
    #'t' are time indices to the values in 'arr'
    # Example: 'dist_arr = time_to_range(t, v, 0, 2)'
    #The output of this funciton can be used to create variables that indicate if an event (e.g. an error) will occur within a certain time range: 'drop_within_60 = np.where(np.logical_and(dist_arr>0, dist_arr<60), 1, 0)'
    [t_sorted, a_sorted] = my_sort([t,arr]) #Sort to help with searching
    dist_arr = np.zeros(len(t_sorted))
    indx = np.where(np.logical_and(a_sorted>=low_value, a_sorted<=high_value))[0]
    indx_prev = 0
    for i in indx:
        dist_arr[indx_prev:i+1] = t_sorted[i]-t_sorted[indx_prev:i+1]
        indx_prev = i+1
    return dist_arr



def dist_to_next_closest(t1, t2, reverse=False):
    #Finds the distance to the timestamp in 't2' that is the closest following in time to each timestamp in 't1'
    #If reverse=True, finds the distance to the timestamp in 't2' that is the closest previous in time to each timestamp in 't1'
    
    dist = np.zeros(t1.shape)
    if reverse==False: 
        t1 = np.sort(t1)
        t2 = np.sort(t2)
        for i, t in enumerate(t1): 
            if t>t2[-1]: break
            while t>t2[0]: t2 = t2[1:]
            dist[i] = t2[0]-t
    else:
        t1 = np.flip(np.sort(t1))
        t2 = np.flip(np.sort(t2))
        for i, t in enumerate(t1): 
            if t<t2[-1]: break
            while t<t2[0]: t2 = t2[1:]
            dist[i] = t-t2[0]
    return dist    



def predictability_metrics(t_faults, t_outliers, t_new, if_plot=False, if_print=False, run_type=0):
    #Finds the predictability metrics for each fault_outlier pair
    #'run_type' is used to return different value for different types of plots if desired
    
    if np.logical_or(len(t_faults)==0, len(t_outliers)==0): return [np.nan,]*21
    
    ### Resample and prep
    dt = t_new[1] - t_new[0]
    t_new_bins = np.concatenate([t_new - dt/2, np.array([t_new[-1]+dt/2])])
    
    #Remove duplicate timestamps
    t_outliers = np.unique(t_outliers)
    t_faults = np.unique(t_faults) 
    
    # Resample and find distance from outliers to next fault
    faults_rs, _ = np.histogram(t_faults, bins=t_new_bins)
    faults_rs = faults_rs>0 #convert to boolean
    faults_rs_dist = dist_to_next_closest(t_new, t_faults) 
    outlier_to_fault = dist_to_next_closest(t_outliers, t_faults)
    
    # Resample and find distance to fault from most recent outlier
    outliers_rs, _ = np.histogram(t_outliers, bins=t_new_bins)
    outliers_rs = outliers_rs>0 #convert to boolean
    outliers_rs_dist = dist_to_next_closest(t_new, t_outliers, reverse=True)
    fault_to_outlier = dist_to_next_closest(t_faults, t_outliers, reverse=True)
    
    #Just for the below plot (a small shift to match faults_rs_dist, not used for results, just plotting)
    outliers_rs2, _ = np.histogram(t_outliers, bins=t_new_bins+dt)
    outliers_rs2 = outliers_rs2>0 #convert to boolean
    x = ((t_new-t_new[0])/60/60/24/7)
    for i, e in enumerate(np.flip(faults_rs_dist)): 
        if e!=0: break
    j = len(faults_rs_dist)-i #where the last fault is (cut off zeros after)
    
    
    ### Figure 6: Seconds to fault with outliers
    if if_plot:
        plt.rcParams.update({'font.size': 10})
        plt.rcParams["font.family"] = "Times New Roman"
        plt.figure(figsize=(3.25, 2), dpi=300)
        plt.plot(x[outliers_rs2][:j], faults_rs_dist[outliers_rs2][:j], '.', color='C1')
        plt.plot(x[:j], faults_rs_dist[:j], color='C0')
        plt.plot(x[outliers_rs2][:j], faults_rs_dist[outliers_rs2][:j], '.', color='C1')
        plt.xlabel('Weeks')
        plt.ylabel('Seconds')
        plt.title('Seconds to fault')
        plt.legend(['Outliers'])
        plt.show() 
    
    if run_type==2: return x[:j], faults_rs_dist[:j], outliers_rs2[:j]
    

    ### Figure 7: Outlier-to-fault and fault-to-oultier CDF
    dt = t_new[1] - t_new[0] #time step length (seconds)
    v_max = np.max([np.max(outlier_to_fault), np.max(fault_to_outlier)]) #max value possible in histograms
    if v_max<3*dt: v_max = 3*dt
    i = np.arange(0,2*v_max, dt)
    x = (i[1:]-i[1]/2)/60/60 #hours
    
    o2f = np.histogram(outlier_to_fault, bins=i)[0].astype(float) #percent outliers that have a fault occur bin distance after
    aaa = np.histogram(faults_rs_dist, bins=i)[0] #total respesentation of each bin
    ii = np.where(aaa!=0)[0] #avoid dividing by zero
    o2f[ii] = o2f[ii]#/aaa[ii]
    o2f = o2f/np.sum(o2f)
    
    f2o = np.histogram(fault_to_outlier, bins=i)[0].astype(float) #percent faults that have an outlier occur bin distance before
    aaa = np.histogram(outliers_rs_dist, bins=i)[0] #total respesentation of each bin
    ii = np.where(aaa!=0)[0] #avoid dividing by zero
    f2o[ii] = f2o[ii]#/aaa[ii]
    f2o = f2o/np.sum(f2o)
        
    if if_plot:
        plt.rcParams.update({'font.size': 10})
        plt.rcParams["font.family"] = "Times New Roman"
        plt.figure(figsize=(3.25, 2), dpi=300)
        plt.plot(x, np.cumsum(o2f), color='C0')
        plt.plot(x, np.cumsum(f2o), '--', color='C1')
        plt.xlabel('Hours')
        plt.ylabel('Distribution')
        plt.title('Cumulative Distribution Function')
        plt.legend(['outlier-to-fault','fault-to-outlier'])
        plt.show()
        
    if run_type==3: return x, o2f, f2o, outlier_to_fault, fault_to_outlier
        
    
    ### Figure 8: Probability of fault per time step
    #Add density to timeline for each outlier
    indxs = np.where(outliers_rs>0)[0]
    c = np.zeros(outliers_rs.shape)
    for j in indxs:
        if (len(c)-j)>len(o2f): offset = len(o2f)
        else: offset = len(c)-j
        c[j:j+offset] += o2f[:offset]
    x = ((t_new-t_new[0])/60/60/24/7)
    
    if if_plot:
        if len(t_faults)>10000: ms = 50000/len(t_faults)
        else: ms = 5
        
        plt.rcParams.update({'font.size': 10})
        plt.rcParams["font.family"] = "Times New Roman"
        plt.figure(figsize=(3.25, 2), dpi=300)
        plt.plot(x[faults_rs][0], c[faults_rs][0], '.', color='C1')
        plt.plot(x, c, color='C0')
        plt.plot(x[faults_rs], c[faults_rs], '.', ms=ms, color='C1')
        plt.xlabel('Weeks')
        plt.ylabel('Probability of fault')
        plt.title('Probability of fault per timestep')
        plt.legend(['Faults'])
        plt.show() 
        
        
    if run_type==1: return x, c, faults_rs
        
        
    ### Figure 9: ROC curve for future fault classification 
    l = []
    tmp = np.linspace(0,1,100)
    for j in tmp:
        
        if j==1: p = c>j #prediction
        else: p = c>=j #prediction
        a = faults_rs #actual
        
        #Calulate classification metrics
        tp = np.sum(np.logical_and(p==1, a==1)) #true positive
        fp = np.sum(np.logical_and(p==1, a==0)) #false positive
        fn = np.sum(np.logical_and(p==0, a==1)) #false negative
        tn = np.sum(np.logical_and(p==0, a==0)) #true negative
        tpr = tp/(tp+fn) #true positive rate (correct positive predictions over actual positive)
        fpr = fp/(fp+tn) #false positive rate (incorrect positive predictions over actual negative)
        tnr = tn/(tn+fp) #true negative rate (correct negative predictions over actual negative)
        fnr = fn/(fn+tp) #false negative rate (incorrect negative predictions over actual postive)
        acc = (tp+tn)/(tp+fp+tn+fn)
        if (tp+fp)==0: prec=0
        else: prec = tp/(tp+fp) #precision (number of correct positive predictions over number of positive predictions)
        rec = tpr #recall
        if (prec+rec)==0: f1=0
        else: f1 = 2*prec*rec/(prec+rec) #F1 score
        
        if (tn+fn)==0: prec_n=0
        else: prec_n = tn/(tn+fn) #precision (number of correct negative predictions over number of negative predictions)
        rec_n = tnr
        if (prec_n+rec_n)==0: f1_n=0
        else: f1_n = 2*prec_n*rec_n/(prec_n+rec_n) #F1 score
        
        if (f1+f1_n)==0: f2=0
        else: f2 = 2*f1*f1_n/(f1+f1_n)
        
        l.append([tpr, fpr, f1, prec, rec, f1_n, f2, tnr, fnr, acc]) #store results
    ll = np.array(l)
    fprs = ll[:,1]
    tprs = ll[:,0]
    
    if if_plot:
        plt.rcParams.update({'font.size': 10})
        plt.rcParams["font.family"] = "Times New Roman"
        plt.figure(figsize=(3.25, 2), dpi=300)
        plt.plot(fprs, tprs, '.-', color='C0')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.show() 
        
    if run_type==4: return fprs, tprs
    
    if if_plot:
        plt.rcParams.update({'font.size': 10})
        plt.rcParams["font.family"] = "Times New Roman"
        plt.figure(figsize=(3.25, 2), dpi=300)
        plt.plot(tmp, ll[:,3], '.-', color='C0')
        plt.plot(tmp, ll[:,4], '.-', color='C1')
        plt.xlabel('Threshold')
        plt.legend(['Precision', 'Recall'])
        plt.show() 
    
    
    ### Calculate and print final metrics
    faults_rnd_std =  np.std(faults_rs_dist)/60/60 #hours, what the STD would be if oultiers randomly distibuted
    oultiers_rnd_std =  np.std(outliers_rs_dist)/60/60 #hours, what the STD would be if faults randomly distributed
    
    a0 = np.mean(outlier_to_fault)/60/60 #hours to fault given an outlier (mean)
    a1 = np.std(outlier_to_fault)/60/60/faults_rnd_std #hours to fault given an outlier (STD) (multiple of random)
    if faults_rnd_std==0: a1 = np.nan #set to np.nan if divided by zero (would be zero otherwise)
    a2 = np.mean(fault_to_outlier)/60/60 #hours from outlier given a fault (mean)
    a3 = np.std(fault_to_outlier)/60/60/oultiers_rnd_std #hours from outlier given a fault (STD) (multiple of random)
    if oultiers_rnd_std==0: a3 = np.nan #set to np.nan if divided by zero (would be zero otherwise)
    a4 = np.max(ll[:,2]) #(np.max(ll[:,2])-f1_min)/(1-f1_min) #best F1
    a5 = np.linspace(0,1,100)[np.argmax(ll[:,2])] #threshold for best F1 score
    tmp = my_sort([fprs, tprs]) #sort list for calculating area under the curve
    a6 = np.trapz(tmp[1], tmp[0]) #AUC
    a7 = np.max(ll[:,6]) #(np.max(ll[:,6])-f2_min)/(1-f2_min) #best F2
    a8 = np.linspace(0,1,100)[np.argmax(ll[:,6])] #threshold for best F2 score
    a9 = np.sum(faults_rs)/len(faults_rs) #positivity rate
    a10 = ll[:,3][np.argmax(ll[:,2])] #precision
    a11 = ll[:,4][np.argmax(ll[:,2])] #recall
    a12 = ll[:,-1][np.argmax(ll[:,2])] #accuracy
    
    p = c>=a5 #prediction
    a = faults_rs #actual
    i_tp = np.where(np.logical_and(p==1, a==1))[0]
    i_fp = np.where(np.logical_and(p==1, a==0))[0]
    i_fn = np.where(np.logical_and(p==0, a==1))[0]
    i_tn = np.where(np.logical_and(p==0, a==0))[0]

    outliers_rs_dist2 = np.flip(dist_to_next_in_range(t_new, np.flip(outliers_rs), low_value=1, high_value=1)) #so faults are zeros instead of just very small like outliers_rs_dist
    a13 = np.mean(outliers_rs_dist2[i_tp])/60/60 #average time true positive predictions are from an outlier (hours)
    a14 = np.std(outliers_rs_dist2[i_tp])/60/60  
    a15 = np.mean(outliers_rs_dist2[i_fp])/60/60 #average time false positive predictions are from an outlier (hours)
    a16 = np.std(outliers_rs_dist2[i_fp])/60/60
    a17 = np.mean(outliers_rs_dist2[i_fn])/60/60 #average time false negative predictions are from an outlier (hours)
    a18 = np.std(outliers_rs_dist2[i_fn])/60/60
    a19 = np.mean(outliers_rs_dist2[i_tn])/60/60 #average time true negative predictions are from an outlier (hours)
    a20 = np.std(outliers_rs_dist2[i_tn])/60/60
    
    if if_print:
        print("Outlier-to-fault (mean, STD multiple of random): %3.3f hrs, %3.3f"%(a0,a1))
        print("Fault-to-outlier (mean, STD multiple of random): %3.3f hrs, %3.3f"%(a2,a3))
        print("F1 score, normalized (best value, best threshold): %3.3f, %3.3f"%(a4,a5))
        print("F2 score, normalized (best value, best threshold): %3.3f, %3.3f"%(a7,a8))
        print("Area Under ROC Curve (AUC): %3.3f"%a6)
    
    return np.array([a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20])