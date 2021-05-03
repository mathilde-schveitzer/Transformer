import csv
import numpy as np
import torch

def get_data2(backast_length, forecast_length, nb, train_set, test_set, device='cpu') :

    
    if len(train_set.shape)>1 :
        assert train_set.shape[0]==test_set.shape[0], 'train and test sets should have same dimension'
        dim=train_set.shape[0]
        ntrain=train_set.shape[1]
        ntest=test_set.shape[1]

        xtrain = np.empty((0, backast_length, dim))
        ytrain = np.empty((0, forecast_length, dim))
    
        xtest = np.empty((0, backast_length, dim))
        ytest = np.empty((0, forecast_length, dim))


        time_series_cleaned_fortraining_x=np.zeros((1, backast_length, dim))
        time_series_cleaned_fortraining_y=np.zeros((1, forecast_length, dim))

        time_series_cleaned_fortesting_x=np.zeros((1, backast_length, dim))
        time_series_cleaned_fortesting_y=np.zeros((1, forecast_length, dim))

        for i in range(nb) : #on selectionne de facon aleatoire nb "bouts de signaux"
            j=np.random.randint(backast_length, ntrain - forecast_length)
            k=np.random.randint(backast_length, ntest - forecast_length)
            time_series_cleaned_fortraining_x=train_set[:,j- backast_length:j].reshape(1,backast_length,dim)
            time_series_cleaned_fortraining_y=train_set[:,j:j+forecast_length].reshape(1,forecast_length, dim)
            time_series_cleaned_fortesting_x=test_set[:,k-backast_length:k].reshape(1,backast_length, dim)
            time_series_cleaned_fortesting_y=test_set[:,k:k+forecast_length].reshape(1,forecast_length,dim)

            xtrain = np.vstack((xtrain, time_series_cleaned_fortraining_x))
            ytrain = np.vstack((ytrain, time_series_cleaned_fortraining_y))
        
            xtest = np.vstack((xtest, time_series_cleaned_fortesting_x))
            ytest = np.vstack((ytest, time_series_cleaned_fortraining_y))

        # print('xtrainshape : ', xtrain.shape)
        # print('ytrainshape : ', ytrain.shape)
        # print('ytestshape : ', ytest.shape)
        # print('xtestshape : ', xtest.shape)

    else :
        ntrain=train_set.shape[0]
        ntest=test_set.shape[0]
        xtrain = np.empty((0, backast_length))
        ytrain = np.empty((0, forecast_length))
    
        xtest = np.empty((0, backast_length))
        ytest = np.empty((0, forecast_length))


        time_series_cleaned_fortraining_x=np.zeros((1, backast_length))
        time_series_cleaned_fortraining_y=np.zeros((1, forecast_length))

        time_series_cleaned_fortesting_x=np.zeros((1, backast_length))
        time_series_cleaned_fortesting_y=np.zeros((1, forecast_length))

        for i in range(nb) : #on selectionne de facon aleatoire nb "bouts de signaux"
            j=np.random.randint(backast_length, ntrain - forecast_length)
            k=np.random.randint(backast_length, ntest - forecast_length)
            time_series_cleaned_fortraining_x=train_set[j- backast_length:j]
            time_series_cleaned_fortraining_y=train_set[j:j+forecast_length]
            time_series_cleaned_fortesting_x=test_set[k-backast_length:k]
            time_series_cleaned_fortesting_y=test_set[k:k+forecast_length]

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
    
def normalize_data(x):
    "value will be btwm 0 and 1"
    min=np.amin(x)
    max=np.amax(x)

    x1=(x-min)/(max-min)
    x2=(x-max)/(max-min)

    return(0.5*(x1+x2))

def merge_data(a,b):
    "return an array which is a concatenate version of a's lines and b's line"
    c=np.zeros(a.shape[0]*(a.shape[1]+b.shape[1]))
    for k in range(a.shape[0]):
        c[k*(a.shape[1]+b.shape[1]):(k+1)*a.shape[1]+k*b.shape[1]]=a[k,:]
        c[(k+1)*(a.shape[1])+k*b.shape[1]:(k+1)*(a.shape[1]+b.shape[1])]=b[k,:]
    return(c)

def merge_line(a,b,k):
            merge=np.zeros(a.shape[1]+b.shape[1])
            merge[:a.shape[1]]=a[k,:]
            merge[a.shape[1]:]=b[k,:]
            return(merge)
