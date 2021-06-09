import sys
import os
import time
import argparse
import torch
from nbeats import *

def main(name,storage,ninp,device='cpu'):

    # you dont need to create a directory for name since it has already been done
    

    name_='transformer_lpos_encod_{}'.format(storage)

    storage_path='./data/{}'.format(name_)
 
    if not os.path.exists(storage_path) :
        os.makedirs(storage_path)

    xtrain=torch.load('./data/{}/xtrain.pt'.format(name))
    ytrain=torch.load('./data/{}/ytrain.pt'.format(name))
    xtest=torch.load('./data/{}/xtest.pt'.format(name))
    ytest=torch.load('./data/{}/ytest.pt'.format(name))
    print(xtrain.shape)
    print(xtest.shape)
    print('ok : we start to load the model')
    
    epochs=500
    bsz=50
    eval_bsz=50
    backcast_length=100 #do not change until you load an other set of data
    forecast_length=100

    ninp+=1


    model=TransformerModel(ninp, nhead=2, nhid=128, nlayers=2, backast_size=backcast_length, forecast_size=forecast_length, pos_encod=True, dropout=0.5, device=device)

    
    print("Model structure: ", model, "\n\n")
    for layer_name, param in model.named_parameters():
        print(f"Layer: {layer_name} | Size: {param.size()} \n")

    start_time=time.time()

    model.fit(xtrain[:,:,:ninp], ytrain[:,:,:ninp], xtest[:,:,:ninp], ytest[:,:,:ninp], bsz, epochs, name_)

    
    elapsed_time=time.time()-start_time
    print('| DL Session took {} seconds |'.format(elapsed_time))    

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

    
