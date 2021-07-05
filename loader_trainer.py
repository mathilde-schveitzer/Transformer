import sys
import os
import time
import argparse
import torch
from nbeats import *

def main(name,storage,ninp,device='cpu'):

    # you dont need to create a directory for name since it has already been done
    

    name_='transformer_t2v_{}'.format(storage)
    name2='transformer_{}'.format(storage)

    storage_path='./data/{}'.format(name_)

    storage_path__='./data/{}'.format(name2)

    if not os.path.exists(storage_path) :
        os.makedirs(storage_path)


    if not os.path.exists(storage_path__) :
        os.makedirs(storage_path__)


    xtrain=torch.load('./data/{}/xtrain.pt'.format(name))
    ytrain=torch.load('./data/{}/ytrain.pt'.format(name))
    xtest=torch.load('./data/{}/xtest.pt'.format(name))
    ytest=torch.load('./data/{}/ytest.pt'.format(name))
    print(xtrain.shape)
    print(xtest.shape)
    print('ok : we start to load the model')
    
    epochs=1000
    bsz=50
    eval_bsz=50
    backcast_length=100 #do not change until you load an other set of data
    forecast_length=100

    ninp+=1

    model=TransformerModel(1, nhead=2, nhid=128, nlayers=2, backast_size=backcast_length, forecast_size=forecast_length, dropout=0.5, t2v=True, device=device)
    model_=TransformerModel(1, nhead=2, nhid=128, nlayers=2, backast_size=backcast_length, forecast_size=forecast_length, dropout=0.5, t2v=False, device=device)

    
    print("Model structure: ", model, "\n\n")
    for layer_name, param in model.named_parameters():
        print(f"Layer: {layer_name} | Size: {param.size()} \n")

    start_time=time.time()
    print(xtrain[:ninp].shape)
    model.fit(xtrain[:,:,ninp-1:ninp], ytrain[:,:,ninp-1:ninp], xtest[:,:,ninp-1:ninp], ytest[:,:,ninp-1:ninp], bsz, epochs, name_)
    elapsed_time=time.time()-start_time
    model_.fit(xtrain[:,:,ninp-1:ninp], ytrain[:,:,ninp-1:ninp], xtest[:,:,ninp-1:ninp], ytest[:,:,ninp-1:ninp], bsz, epochs, name2)
    elapsed_time_=time.time()-elapsed_time
    
   
    print('| DL Session took {} seconds with T2V and {} seconds without |'.format(elapsed_time, elapsed_time_))    

    print('---------- Name of the file : {} --------------'.format(storage))


    
if __name__ == '__main__':
   parser=argparse.ArgumentParser()
   parser.add_argument('data', help='The name of the folder in which you want to load the data from')
   parser.add_argument('storage', help='The name of the folder in which out data will be saved')
   parser.add_argument('ninp', help='number of input signals')
   parser.add_argument('-device', help='Processor used for torch')
   args=parser.parse_args()
   if args.device :
       main(args.data, args.storage, int(args.ninp), device=args.device)
   else :
       main(args.data, args.storage, int(args.ninp))

    
