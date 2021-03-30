import csv
import numpy as np
import torch

def get_data(filename, batch_size=200, eval_batch_size=100, device='cpu'):
    """
    Create the set of datas that will be used to train the neural networks, from the file filename
    
    Args :
         - filename : string
               Name of the file that contains the signal, must be a .csv
         - copy : int
               After getting the signal, the algo copies it copy times in a numpy array. Each copy corresponds to a sample(x,y)
"""
    #clean time_series :
    def clean_time_series(filename):
        x_tl = []
        name='./data/{}/signal.csv'.format(filename)
        with open(name, "r") as file:
            reader = csv.reader(file, delimiter=',')
            for line in reader:
                x_tl.append(line)
        x_tl=x_tl[0] #petite astuce : x_tl est en faite une ligne dans une ligne
        time_series = [float(s) for s in x_tl if s != '']
        print('---------- time_series len = {} ------------'.format(len(time_series)))
        return(time_series)

    cleaned_time_series=clean_time_series(filename)
    n=len(cleaned_time_series)
    ntest=int(0.75*n)
    nvalid=n # no validation test for now

    xtrain=torch.tensor(np.array(cleaned_time_series[:ntest]), dtype=torch.float32).to(device)
    xtest=torch.tensor(np.array(cleaned_time_series[ntest:nvalid]),dtype=torch.float32).to(device)
    xval=torch.tensor(np.array(cleaned_time_series[nvalid:]),dtype=torch.float32).to(device)
    
    # at this stage : 1 line of size npts

    def batchify(data, bsz):
        # Divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz) 
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data

    train_data=batchify(xtrain, batch_size)
    val_data= batchify(xval, eval_batch_size)
    test_data = batchify(xtest, eval_batch_size)
    # train_data.shape --> nb_batch, batch_size
    
    return(train_data,test_data,val_data)

def get_batch(source,i,bptt,printer=False):
    seq_len=min(bptt, len(source)-(i+1))
    data=source[i:i+seq_len]
    if printer :
        print('--------------get_batch--------------------------------------')
        print(seq_len,source.size(1),1)
        print(data.shape)
    data=torch.reshape(data,(seq_len,source.size(1),1))
    target=source[i+1:i+1+seq_len].reshape(-1) # on perd une dimension
        
    return data, target # en sortie on a bptt donnees, de longueurs batch_size

def squeeze_last_dim(tensor): # apply before reshape
    if len(tensor.shape)==3 and tensor.shape[-1]==1 :
        return tensor[..., 0]
    return tensor
