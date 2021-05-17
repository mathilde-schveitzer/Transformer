import csv
from tqdm import tqdm
import numpy as np
import torch
import glob

def get_data2(backast_length, forecast_length, n_interval, train_set, test_set) :

    
    assert len(train_set.shape)>1, 'please squeeze your data'
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

    length=forecast_length+backast_length

    
    for i in tqdm(range(0,ntrain-length-n_interval*10,backast_length)) :
        for gap in range(0,n_interval*10,10) :
            print(i,gap)
            time_series_cleaned_fortraining_x=train_set[:,i+gap:i+gap+backast_length].reshape(1,dim,backast_length).swapaxes(1,2)
            time_series_cleaned_fortraining_y=train_set[:,(i+gap+backast_length):(i+gap+backast_length+forecast_length)].reshape(1,dim,forecast_length).swapaxes(1,2)

            xtrain = np.vstack((xtrain, time_series_cleaned_fortraining_x))
            ytrain = np.vstack((ytrain, time_series_cleaned_fortraining_y))

    for k in tqdm(range(0,ntest-length-n_interval*10,backast_length)):
        for gap in range(0,n_interval*10,10) : #multiplie par n_interval le nb de donnees utilisables
            time_series_cleaned_fortesting_x=test_set[:,k+gap:k+gap+backast_length].reshape(1, dim, backast_length).swapaxes(1,2)
            time_series_cleaned_fortesting_y=test_set[:,k+gap+backast_length:(k+gap+forecast_length+backast_length)].reshape(1, dim, forecast_length).swapaxes(1,2)

            xtest = np.vstack((xtest, time_series_cleaned_fortesting_x))
            ytest = np.vstack((ytest, time_series_cleaned_fortraining_y))

    xtrain,ytrain=shuffle_in_unison(xtrain, ytrain)
    xtest,ytest=shuffle_in_unison(xtest, ytest)

    xtrain=torch.tensor(xtrain,dtype=torch.float32)
    ytrain=torch.tensor(ytrain,dtype=torch.float32)
    xtest=torch.tensor(xtest,dtype=torch.float32)
    ytest=torch.tensor(ytest, dtype=torch.float32)

    
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

        
def get_data_for_predict(backast_length, data_set) :

    
    if len(data_set.shape)>1 :
       
        dim=data_set.shape[0]
        n=data_set.shape[1]
       

        data_train = np.empty((0, backast_length, dim))

        time_series_cleaned_for_predicting=np.zeros((1, backast_length, dim))
       
        for i in range(0,n,backast_length) : #on passe en revue le signal, dans l'ordre et en conservant le format utilise lors des autres sessions
            time_series_cleaned_for_predicting=data_set[:,i:i+backast_length].reshape(1,dim,backast_length).swapaxes(1,2)
           
            data_train = np.vstack((data_train, time_series_cleaned_for_predicting))

    else :
        n=data_set.shape[0]
       
        data_train = np.empty((0, backast_length))
        time_series_cleaned_for_prediction=np.zeros((1, backast_length))
       
        for i in range(0,n,backast_length) : #on selectionne de facon aleatoire nb "bouts de signaux"
            time_series_cleaned_for_predicting=train_set[i:i+backast_length]
            data_train = np.vstack((data_train, time_series_cleaned_for_predicting))

    print('data_train.shape : ', data_train.shape)

    data_train=torch.tensor(data_train,dtype=torch.float32)

                                
    return data_train    
    
def read_signal(filename) :
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

def register_training_signal(nlimit) :

    #load only the id column
    
    path='nbeats_f100/train/SAT2_10_minutes_future100_*.csv'
    files=glob.glob(path)
    x=np.empty
    for k,filename in enumerate(files):
        if k==0 :
            time_series=read_signal(filename)
            x=time_series[:,:nlimit+1]
        else :
            time_series=read_signal(filename)
            x=np.vstack((x,time_series[:,:nlimit+1]))
    return(x)
        
def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return(a,b)
