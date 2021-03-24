import torch
import torch.nn as nn
import time
import argparse
import generate_signal as gs
from load_data import get_data
from model import *

def main(name,device):

   # we generate the signal which will be analyzed
   length_seconds, sampling_rate=10000, 150 #that makes 15000 pts
   freq_list=[0.05,0.5,5]
   print('----creating the signal, plz wait------')
   sig=gs.generate_signal(length_seconds, sampling_rate, freq_list)
   print('finish : we start storing it in a csv file')
   path='./data/{}'.format(name)
   if not os.path.exists(path) :
      os.makedirs(path)    
   gs.register_signal(sig[0],'./data/{}/signal'.format(name))
   print('----we got it : time to create the ndarray-----')

   train_set,test_set,validation_set=get_data(name,device=device)
   
   print('--------------- we got the data -------------')
   print(train_set)
   print('-----------train_set.shape()--------')
   print(train_set.shape)
   print(train_set.size)
   
   #Initiate an instance :
   ninp=1
   nhid=200
   nlayers=2
   nhead=3
   dropout=0.2
   bptt=10
   ntokens=12
      
   model=TransformerModel(ntokens, ninp, nhead, nhid, nlayers, bptt, dropout,device=device)

   epochs=200
   model.fit(train_set, validation_set, epochs, name)
      
   #Evaluate the model with the test dataset

   test_loss = model.evaluate(test_set, val=True)
   train_loss = model.evaluate(train_set, val=False)
   print('=' * 89)
   print('| End of training | test loss {:5.2f} | train loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, train_loss, math.exp(test_loss)))
   print('=' * 89)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('name', help='Name of the signal you will create')
    parser.add_argument('device', help='Processor used for torch tensor')
    args=parser.parse_args()
    main(args.name,args.device)
