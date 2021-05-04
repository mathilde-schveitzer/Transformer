import csv
import os
import argparse
import numpy as np
import sys
import glob
import tqdm as tqdm
from load_data import normalize_data

def register_signal(filename) :
    with open(filename, newline='') as csvfile :
         reader=csv.reader(csvfile, delimiter=',')

         for n,row in enumerate(reader) :
             if n==0 :
                 length=len(row)
                 x=np.zeros((0,length-1))
             else :
                 time_series=np.zeros((1,length-1))
                 for k,column in enumerate(row) :
                     if not(k==0) : #switch first colomn (time step)
                         time_series[0,k-1]=column
                    
                 x=np.vstack((x,time_series))
    return(x)

def normalize_data(x):
    "value will be btwm 0 and 1"
    min=np.amin(x)
    max=np.amax(x)

    x1=(x-min)/(max-min)
    x2=(x-max)/(max-min)

    return(0.5*(x1+x2))

def pretreatment(id):
    
    time_series=np.zeros((1,2))
    path="/data1/infantes/kratos/raw_data/dataset_2/SAT2/TM{}_*.csv".format(id)
    files=glob.glob(path)

   
    for i in range(len(files)):
        
        namecsv=files[i].split('/')
        namecsv=namecsv[-1]
        name=namecsv.split('.')
        name=name[0]
        x=np.zeros((0,2))
        print(files[i])
        print('--------------file n* {} // {} -------'.format(i,len(files)))
        with open(files[i], newline='') as csvfile :
             reader=csv.reader(csvfile, delimiter=',')
             for n,row in enumerate(reader) :
                 if n>0 : #skip first line
                     for k,column in enumerate(row) :
                         if k==1 : # Time
                             time_series[0,0]=column
                         if k==3 :
                             time_series[0,1]=column
    
                     x=np.vstack((x,time_series))
                     if n % 1000 == 0 :
                         print(x.shape)
                    
        np.savetxt('./data/{}.txt'.format(name),x)

        return(name)
    
def from_txt_file_filter(x):
    y=np.zeros((0,2))
    alpha=10**(-2)
    for k in range(x.shape[0]-1) :
        if abs(x[k,1]-x[k+1,1])>=alpha :
            y=np.vstack((y,x[k,:]))
            print(y.shape)
    return(y)
    
