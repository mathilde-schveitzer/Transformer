import csv
import numpy as np
import torch

# def get_data(filename, batch_size=200, eval_batch_size=100, device='cpu'):
#     """
#     Create the set of datas that will be used to train the neural networks, from the file filename
    
#     Args :
#          - filename : string
#                Name of the file that contains the signal, must be a .csv
#          - copy : int
#                After getting the signal, the algo copies it copy times in a numpy array. Each copy corresponds to a sample(x,y)
# """
#     #clean time_series :
#     def clean_time_series(filename):
#         x_tl = []
#         name='./data/{}/signal.csv'.format(filename)
#         with open(name, "r") as file:
#             reader = csv.reader(file, delimiter=',')
#             for line in reader:
#                 x_tl.append(line)
#         cleaned_serie=np.zeros((len(x_tl),len(x_tl[0])
#         for k, xtl in enumerate(x_tl) :
#             time_series = [float(s) for s in xtl if s != '']
#             cleaned_series[k,:]=time_series
#         return(cleaned_series)

#     cleaned_time_series=clean_time_series(filename)
#     n=len(cleaned_time_series)
#     ntest=int(0.75*n)
#     nvalid=n # no validation test for now

#     xtrain=torch.tensor(np.array(cleaned_time_series[:,:ntest]), dtype=torch.float32).to(device)
#     xtest=torch.tensor(np.array(cleaned_time_series[:,ntest:nvalid]),dtype=torch.float32).to(device)
#     xval=torch.tensor(np.array(cleaned_time_series[:,nvalid:]),dtype=torch.float32).to(device)
    
#     # at this stage : k line of size npts

#     def batchify(data, bsz):

#         # Divide the dataset into bsz parts.
#         if len(data.shape)>1 :
#             nbatch = data.shape[1] // bsz
#         # Trim off any extra elements that wouldn't cleanly fit (remainders).
#         else :
#             nbatch = data.shape[0] // bsz
#             data = data.narrow(0, 0, nbatch * bsz) 
#         # Evenly divide the data across the bsz batches.
#         data = data.view(bsz, -1).t().contiguous()
#         return data

#     train_data=batchify(xtrain, batch_size)
#     val_data= batchify(xval, eval_batch_size)
#     test_data = batchify(xtest, eval_batch_size)
#     # train_data.shape --> nb_batch, batch_size
    
#     return(train_data,test_data,val_data)

# def get_batch(source,i,bptt,printer=False):
#     assert i+backast_size+forecast_size>
#                                 seq_len=min(bptt, len(source)-(i+1))
#     data=source[i:i+seq_len]
#     data=torch.reshape(data,(seq_len,source.size(1),1))
#     target=source[i+1:i+1+seq_len].reshape(-1) # on perd une dimension
        
#     return data, target # en sortie on a bptt donnees, de longueurs batch_size

# def squeeze_last_dim(tensor): # apply before reshape
#     if len(tensor.shape)==3 and tensor.shape[-1]==1 :
#         return tensor[..., 0]
#     return tensor

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
    
    
                               
