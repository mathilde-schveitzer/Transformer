import torch
import torch.nn as nn
import time
import argparse
from model import *
import generate_signal as gs
import torch.nn as nn
from load_data import get_data

def main(name,device):

   # we generate the signal which will be analyzed
   length_seconds, sampling_rate=1000, 150 #that makes 15000 pts
   freq_list=[0.5]
   print('----creating the signal, plz wait------')
   sig=gs.generate_signal(length_seconds, sampling_rate, freq_list)
   print('finish : we start storing it in a csv file')
   gs.register_signal(sig[0],'./data/{}'.format(name))
   print('----we got it : time to create the ndarray-----')

   train_set,_,validation_set=get_data(name,device=device)

   print('--------------- we got the data -------------')
   
   #Initiate an instance :
   emsize=200
   nhid=200
   nlayers=2
   nhead=2
   dropout=0.2
   bptt=1
   ntokens=12
      
   model=TransformerModel(ntokens, emsize, nhead, nhid, nlayers, bptt, dropout).to(device)

   model.criterion= nn.CrossEntropyLoss()
   model.scheduler = torch.optim.lr_scheduler.StepLR(model.optimizer, 1.0, gamma=0.95)
   epochs=3

   best_model, best_val_loss=model.fit(train_set, validation_set, epochs)
      
   #Evaluate the model with the test dataset

   test_loss = evaluate(best_model, test_data)
   print('=' * 89)
   print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, math.exp(test_loss)))
   print('=' * 89)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('name', help='Name of the signal you will create')
    parser.add_argument('device', help='Processor used for torch tensor')
    args=parser.parse_args()
    main(args.name,args.device)
