import torch 
import torch.nn as nn
import time
import argparse
import generate_signal as gs
from load_data import get_data
from model import *

def main(name,device):
   
   name_e=name+'_encod'
   name_ne=name+'_no_encod'
   
   # we generate the signal which will be analyzed
   path_e='./data/{}'.format(name_e)
   path_ne='./data/{}'.format(name_ne)

   sig=np.arange(15000)

   if not os.path.exists(path_e) :
      os.makedirs(path_e)    

   gs.register_signal(sig,'./data/{}/signal'.format(name_e))


   if not os.path.exists(path_ne) :
      os.makedirs(path_ne)    

   gs.register_signal(sig,'./data/{}/signal'.format(name_ne))

   print('----we got it : time to create the ndarray-----')
   
   train_set,test_set,validation_set=get_data(name_e, batch_size=140, eval_batch_size=140, device=device)
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
      


   model_encod=TransformerModel(ninp, nhead, nhid, nlayers, nMLP, bptt, pos_encod=True, dropout=dropout, device=device)
   model_no_encod=TransformerModel(ninp, nhead, nhid, nlayers, nMLP, bptt, pos_encod=False, dropout=dropout, device=device)

   epochs=50000
   
   name_e=name+'_encod'
   model_encod.fit(train_set, test_set, epochs, name_e)

   name_ne=name+'_no_encod'
   model_no_encod.fit(train_set, test_set, epochs, name_ne)
   #Evaluate the model with the test dataset

   test_loss_encod = model_encod.evaluate(test_set, val=True)
   train_loss_encod = model_encod.evaluate(train_set, val=False, predict=True)
   print('=' * 89)
   print('| End of training | test loss encod {:5.2f} | train loss encod {:5.2f} '.format(test_loss_encod, train_loss_encod))
   print('=' * 89)
   test_loss_no_encod = model_no_encod.evaluate(test_set, val=True)
   train_loss_no_encod = model_no_encod.evaluate(train_set, val=False, predict=True)
   print('=' * 89)
   print('| End of training | test loss without encod {:5.2f} | train loss without encod {:5.2f} | '.format(test_loss_no_encod, train_loss_no_encod))
   print('=' * 89)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('name', help='Name of the signal you will create')
    parser.add_argument('device', help='Processor used for torch tensor')
    args=parser.parse_args()
    main(args.name,args.device)
