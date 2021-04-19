import csv
import numpy as np
import torch

def get_data2(backast_length, forecast_length, nb, name, device='cpu') :

    filename='./data/{}/signal.txt'.format(name)
    cleaned_time_series=np.loadtxt(filename)
    if len(cleaned_time_series>1) :
        dim=cleaned_time_series.shape[0]
        length=cleaned_time_series.shape[1]
    else :
        length=cleaned_time_series.shape[0]
    n=length
    ntest=int(0.75*n)
    xtrain = np.empty((0, backast_length, dim))
    ytrain = np.empty((0, forecast_length, dim))
    
    xtest = np.empty((0, backast_length, dim))
    ytest = np.empty((0, forecast_length, dim))


    time_series_cleaned_fortraining_x=np.zeros((1, backast_length, dim))
    time_series_cleaned_fortraining_y=np.zeros((1, forecast_length, dim))

    time_series_cleaned_fortesting_x=np.zeros((1, backast_length, dim))
    time_series_cleaned_fortesting_y=np.zeros((1, forecast_length, dim))

    for i in range(nb) :
        j=np.random.randint(backast_length, ntest - forecast_length)
        k=np.random.randint(ntest+backast_length, n-forecast_length)
        time_series_cleaned_fortraining_x=cleaned_time_series[:,j- backast_length:j].reshape(1,backast_length,dim)
        time_series_cleaned_fortraining_y=cleaned_time_series[:,j:j+forecast_length].reshape(1,forecast_length, dim)
        time_series_cleaned_fortesting_x=cleaned_time_series[:,k-backast_length:k].reshape(1,backast_length, dim)
        time_series_cleaned_fortesting_y=cleaned_time_series[:,k:k+forecast_length].reshape(1,forecast_length,dim)

        xtrain = np.vstack((xtrain, time_series_cleaned_fortraining_x))
        ytrain = np.vstack((ytrain, time_series_cleaned_fortraining_y))
        
        xtest = np.vstack((xtest, time_series_cleaned_fortesting_x))
        ytest = np.vstack((ytest, time_series_cleaned_fortraining_y))

        print('xtrainshape : ', xtrain.shape)
        print('ytrainshape : ', ytrain.shape)
        print('ytestshape : ', ytest.shape)
        print('xtestshape : ', xtest.shape)

    xtrain=torch.tensor(xtrain,dtype=torch.float32).to(device)
    ytrain=torch.tensor(ytrain,dtype=torch.float32).to(device)
    xtest=torch.tensor(xtest,dtype=torch.float32).to(device)
    ytest=torch.tensor(ytest, dtype=torch.float32).to(device)

    print('xtrainshape : ', xtrain.shape)
    print('ytrainshape : ', ytrain.shape)
    print('ytestshape : ', ytest.shape)
    print('xtestshape : ', xtest.shape)
                                
    return xtrain, ytrain, xtest, ytest

def get_xi(xtrain,dim,i):
    "return the x multidimensionnal vector numero i"
    return(xtrain[i*dim:i*(dim+1),:])

def get_yi(ytrain,dim,i):
    return(ytrain[i*dim:i*(dim+1),:])

def batchify(x,y,bsz,i,dim) :
    data=torch.tensor([]).reshape(x.shape(1))
    target=torch.tensor([]).reshape(y.shape(1))
    for k in range(i,i+bsz) :
        data=torch.stack(data, get_xi(x,dim,k))
        target=torch.stack(target, get_yi(y,dim,k))
    print(data.shape)
    print(target.shape)
    return data, target
    
    
                               
