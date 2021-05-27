import os
import time
import argparse
import torch
from nbeats import *

def main(name,storage,device='cpu'):

    # you dont need to create a directory for name since it has already been done
    storage_path='./data/{}'.format(name)
    
    if not os.path.exists(storage_path) :
        os.makedirs(storage_path)

    xtrain=torch.load('./data/{}/xtrain.pt'.format(name))
    ytrain=torch.load('./data/{}/ytrain.pt'.format(name))
    xtest=torch.load('./data/{}/xtest.pt'.format(name))
    ytest=torch.load('./data/{}/ytest.pt'.format(name))

    print('ok : we start to load the model')
    
    epochs=200
    bsz=256
    eval_bsz=256
    backcast_length=100 #do not change until you load an other set of data
    forecast_length=100
    
    model=NBeatsNet(device=device, forecast_length=forecast_length, backcast_length=backcast_length)
    
    print("Model structure: ", model, "\n\n")
    for layer_name, param in model.named_parameters():
        print(f"Layer: {layer_name} | Size: {param.size()} \n")

    start_time=time.time()
    model.fit(xtrain, ytrain, xtest, ytest, storage, epochs=epochs, batch_size=bsz)
    elapsed_time=time.time()-start_time
    test_loss = model.evaluate(xtest, ytest, eval_bsz, storage, False, save=True)
    train_loss = model.evaluate(xtrain, ytrain, bsz, storage, True, save=True)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | train loss {:5.2f} | '.format(test_loss, train_loss))
    print('=' * 89)
    print('| DL Session took {} seconds |'.format(elapsed_time))    

    print('---------- Name of the file : {} --------------'.format(storage))


    
if __name__ == '__main__':
   parser=argparse.ArgumentParser()
   parser.add_argument('data', help='The name of the folder in which you want to load the data from')
   parser.add_argument('storage', help='The name of the folder in which out data will be saved')
   parser.add_argument('-device', help='Processor used for torch')
   args=parser.parse_args()
   if args.device :
       main(args.data, args.storage, device=args.device)
   else :
       main(args.data, args.storage)

    
