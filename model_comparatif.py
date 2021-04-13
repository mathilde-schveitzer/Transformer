import torch
import torch.nn as nn
import time
import argparse
import generate_signal as gs
from load_data import get_data_for_TM
from data import get_data_for_NB

from TM_model import *
from NB_model import *

def main(name,device):

   # we generate the signal which will be analyzed
   length_seconds, sampling_rate=10000, 150 #that makes 15000 pts
   freq_list=[0.05,0.5,0.2,0.4,5,0.001]
   print('----creating the signal, plz wait------')
   sig=gs.generate_signal(length_seconds, sampling_rate, freq_list, add_noise=True)
   print('finish : we start storing it in a csv file')
   path='./data/{}'.format(name)
   if not os.path.exists(path) :
      os.makedirs(path)    
   gs.register_signal(sig[0],'./data/{}/signal'.format(name))
   print('----we got it : time to create the ndarray-----')

   train_set,test_set,_=get_data_for_TM(name, batch_size=540, eval_batch_size=540, device=device)

   backcast_length=100
   forecast_length = backcast_length #convention : pour matcher avec Transfomer et bptt

   xtrain, ytrain, xtest, ytest=get_data_for_NB(name, backcast_length, forecast_length,limit=int(length_seconds*sampling_rate*0.9))

   
   print('--------------- we got the data -------------')
  
   #Initiate an instance :
   ninp=1
   nhid=240
   nlayers=1
   nMLP=124
   nhead=8
   dropout=0.1
   bptt=forecast_length
   batch_size=540
   eval_batch_size=540

   model_TM=TransformerModel(ninp, nhead, nhid, nlayers, nMLP, bptt, dropout=dropout, device=device)

   epochs=2000
   
   model_TM.fit(train_set, test_set, epochs, name)

   model_NB= NBeatsNet(backcast_length=backcast_length, forecast_length=forecast_length, stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK), nb_blocks_per_stack=2, thetas_dim=(4,4), share_weights_in_stack=True, hidden_layer_units=64)
   model_NB.compile_model(loss='mae', learning_rate=1e-2)
   model_NB.fit(name, xtrain, ytrain, validation_data=(xtest,ytest), epochs=epochs, batch_size=batch_size)

   

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('name', help='Name of the signal you will create')
    parser.add_argument('device', help='Processor used for torch tensor')
    args=parser.parse_args()
    main(args.name,args.device)
