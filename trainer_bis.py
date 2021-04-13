import torch
import torch.nn as nn
import time
import argparse
import generate_signal as gs
from load_data import get_data_for_TM
from TM_model import *

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
   print('--------------- we got the data -------------')
  
   #Initiate an instance :
   ninp=1
   nhid=240
   nlayers=1
   nMLP=124
   nhead=15
   dropout=0.1
   bptt=100

   model_encod=TransformerModel(ninp, nhead, nhid, nlayers, nMLP, bptt, pos_encod=True, dropout=dropout, device=device)
   model_no_encod=TransformerModel(ninp, nhead, nhid, nlayers, nMLP, bptt, pos_encod=False, dropout=dropout, device=device)
   
   epochs=500

   name_e=name+'_encod'
   model_encod.fit(train_set, test_set, epochs, name_e)

   name_ne=name+'_no_encod'
   model_no_encod.fit(train_set, test_set, epochs, name_ne)
   #Evaluate the model with the test dataset

   test_loss_encod = model_encod.evaluate(test_set, val=True)
   train_loss_encod = model_encod.evaluate(train_set, val=False, predict=True)
   print('=' * 89)
   print('| End of training | test loss encod {:5.2f} | train loss encod {:5.2f} | test ppl {:8.2f}'.format(test_loss_encod, train_loss_encod, math.exp(test_loss)))
   print('=' * 89)
   test_loss_no_encod = model_no_encod.evaluate(test_set, val=True)
   train_loss_no_encod = model_no_encod.evaluate(train_set, val=False, predict=True)
   print('=' * 89)
   print('| End of training | test loss without encod {:5.2f} | train loss without encod {:5.2f} | test ppl {:8.2f}'.format(test_loss_no_encod, train_loss_no_encod, math.exp(test_loss)))
   print('=' * 89)



if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('name', help='Name of the signal you will create')
    parser.add_argument('device', help='Processor used for torch tensor')
    args=parser.parse_args()
    main(args.name,args.device)
