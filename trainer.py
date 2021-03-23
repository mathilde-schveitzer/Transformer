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
   freq_list=[0.02,0.03,0.5,0.4,5,4,8,9,10]
   print('----creating the signal, plz wait------')
   sig=gs.generate_signal(length_seconds, sampling_rate, freq_list)
   print('finish : we start storing it in a csv file')
   gs.register_signal(sig[0],'./data/{}'.format(name))
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
   nlayers=4
   nhead=1
   dropout=0.2
   bptt=100
   ntokens=12
      
   model=TransformerModel(ntokens, ninp, nhead, nhid, nlayers, bptt, dropout).to(device)

#   lr=5
#  model.scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, 1.0, gamma=0.95)

   epochs=1000
   best_model, best_val_loss=model.fit(train_set, validation_set, epochs)
      
   #Evaluate the model with the test dataset

   test_loss = best_model.evaluate(test_set, val=True)
   train_loss = best_model.evaluate(train_set, val=False)
   print('=' * 89)
   print('| End of training | test loss {:5.2f} | train loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, train_loss, math.exp(test_loss)))
   print('=' * 89)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('name', help='Name of the signal you will create')
    parser.add_argument('device', help='Processor used for torch tensor')
    args=parser.parse_args()
    main(args.name,args.device)
