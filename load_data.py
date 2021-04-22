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
    return data, target
    
    
