import torch 
import torch.nn as nn
import time
import argparse
import generate_signal as gs
from load_data import get_data
from model import *

def main(name,device):

   # we generate the signal which will be analyzed
   path='./data/{}'.format(name)
   sig=np.arange(200)
   if not os.path.exists(path) :
      os.makedirs(path)    
   gs.register_signal(sig,'./data/{}/signal'.format(name))
   print('----we got it : time to create the ndarray-----')

   train_set,test_set,validation_set=get_data(name, batch_size=12, eval_batch_size=12, device=device)
   print('--------------- we got the data -------------')
   print(train_set)
   print('-----------train_set.shape()--------')
   print(train_set.shape)
   print(train_set.size)
   print('-----------test_set.shape()--------')
   print(test_set.shape)
   print('------------val_set.shape()-------')
   print(validation_set.shape)
   
   #Initiate an instance :
   ninp=1
   nhid=240
   nlayers=1
   nhead=4
   nMLP=124
   dropout=0.05
   bptt=3
      
   model=TransformerModel(ninp, nhead, nhid, nlayers, nMLP, bptt, dropout,device=device)

   epochs=2
   model.fit(train_set, validation_set, epochs, name)
      
   #Evaluate the model with the test dataset

   test_loss = model.evaluate(test_set, val=True)
   train_loss = model.evaluate(train_set, val=False, predict=True)
   print('=' * 89)
   print('| End of training | test loss {:5.2f} | train loss {:5.2f} |'.format(test_loss, train_loss))
   print('=' * 89)

   print('---------- Name of the file : {} --------------'.format(name))

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('name', help='Name of the signal you will create')
    parser.add_argument('device', help='Processor used for torch tensor')
    args=parser.parse_args()
    main(args.name,args.device)
