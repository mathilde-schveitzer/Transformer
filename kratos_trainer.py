import sys
import torch
import torch.nn as nn
import time
import argparse
from load_data import *
from nbeats import *
import os

def main(name,device='cpu'):

    backcast_length=80
    forecast_length=80
    ninterval=backcast_length//10

    xtrain, ytrain, xtest, ytest = get_all_data(backcast_length, forecast_length, ninterval, name)

    print('we got the data : xtrain.shape :', xtrain.shape)
    print(ytrain.shape)
    print(xtest.shape)
    print(ytest.shape)
    
    torch.save(xtrain,'./data/{}/xtrain.pt'.format(name))
    torch.save(ytrain,'./data/{}/ytrain.pt'.format(name))
    torch.save(ytest,'./data/{}/ytest.pt'.format(name))
    torch.save(xtest,'./data/{}/xtest.pt'.format(name))

    
    #Initiate an instance :
    
    epochs=1000
    bsz=128
    eval_bsz=128
   
    model=NBeatsNet(device=device, forecast_length=forecast_length, backcast_length=backast_length)
    
    print("Model structure: ", model, "\n\n")
    for layer_name, param in model.named_parameters():
        print(f"Layer: {layer_name} | Size: {param.size()} \n")

    start_time=time.time()
    model.fit(xtrain, ytrain, xtest, ytest, name, epochs=epochs, batch_size=bsz)
    elapsed_time=time.time()-start_time
    test_loss = model.evaluate(xtest, ytest, eval_bsz, name, False, save=True)
    train_loss = model.evaluate(xtrain, ytrain, bsz, name, True, save=True)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | train loss {:5.2f} | '.format(test_loss, train_loss))
    print('=' * 89)
    print('| DL Session took {} seconds |'.format(elapsed_time))    

    print('---------- Name of the file : {} --------------'.format(name))


    
if __name__ == '__main__':
   parser=argparse.ArgumentParser()
   parser.add_argument('name', help='The name of the folder in which out data will be saved')
   parser.add_argument('-device', help='Processor used for torch')
   args=parser.parse_args()
   if args.device :
       main(args.name, device=args.device)
   else :
       main(args.name)
