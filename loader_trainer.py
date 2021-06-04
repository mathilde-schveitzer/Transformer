import sys
import os
import time
import argparse
import torch
from nbeats import *

def main(name,storage,ninp,device='cpu'):

    # you dont need to create a directory for name since it has already been done
    

    name_='nbeats_tr_{}'.format(storage)
  #  name1='nbeats_fc_{}'.format(storage)
  #  name2='transformer_{}'.format(storage)

    storage_path='./data/{}'.format(name_)
 #   storage_path_='./data/{}'.format(name1)
 #   storage_path__='./data/{}'.format(name2)

    if not os.path.exists(storage_path) :
        os.makedirs(storage_path)

  #  if not os.path.exists(storage_path_) :
  #      os.makedirs(storage_path_)

  #  if not os.path.exists(storage_path__) :
  #      os.makedirs(storage_path__)


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

#    model=NBeatsNet(ninp, device=device, forecast_length=forecast_length, backcast_length=backcast_length,block_type='fully_connected')
    model__=NBeatsNet(ninp, device=device, forecast_length=forecast_length, backcast_length=backcast_length,block_type='Tr')
#    model_=TransformerModel(ninp, nhead=2, nhid=128, nlayers=2, backast_size=backcast_length, forecast_size=forecast_length, dropout=0.5, device=device)

    
    print("Model structure: ", model__, "\n\n")
    for layer_name, param in model__.named_parameters():
        print(f"Layer: {layer_name} | Size: {param.size()} \n")

    start_time=time.time()
 #   model.fit(xtrain, ytrain, xtest, ytest, name1, epochs=epochs, batch_size=bsz)
    model__.fit(xtrain, ytrain, xtest, ytest, name_, epochs=epochs, batch_size=bsz)
 #   model_.fit(xtrain, ytrain, xtest, ytest, bsz, epochs, name2)
    
    elapsed_time=time.time()-start_time
 #   test_loss = model.evaluate(xtest, ytest, eval_bsz, storage, False, save=False)
 #   train_loss = model.evaluate(xtrain, ytrain, bsz, storage, True, save=False)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | train loss {:5.2f} | '.format(test_loss, train_loss))
    print('=' * 89)
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

    
