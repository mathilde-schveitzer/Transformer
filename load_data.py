import csv
import numpy as np
import torch

def get_data(filename, batch_size=20, eval_batch_size=10, device='cpu'):
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
        name='{}'.format(filename)
        with open(name, "r") as file:
            reader = csv.reader(file, delimiter=',')
            for line in reader:
                x_tl.append(line)
            time_series = np.array(x_tl)
            time_series = [float(s) for s in time_series if s != '']
        print('---------- time_series len = {} ------------'.format(len(time_series)))
        return(time_series)

    cleaned_time_series=clean_time_series(filename)
    n=len(cleaned_time_series)
    ntest=int(0.75*n)
    nvalid=int(0.875*n)

    xtrain=torch.from_numpy(np.array(cleaned_time_series[:ntest])).to(device)
    xtest=torch.from_numpy(np.array(cleaned_time_series[ntest:nvalid])).to(device)
    xval=torch.from_numpy(np.array(cleaned_time_series[nvalid:])).to(device)

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

    return(train_data,test_data,val_data)
