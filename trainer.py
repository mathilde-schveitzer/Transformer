import torch
import torch.nn as nn
import time
import argparse
import generate_signal as gs
from load_data import get_data
from model import *

def main(name,device,bptt=100):

   name_=name+'_encod_{}'.format(bptt)
   name__=name+'_no_encod_{}'.format(bptt)

   # we generate the signal which will be analyzed
   length_seconds, sampling_rate=10000, 150 #that makes 15000 pts
   freq_list=[0.05,0.5,0.2,0.4,5,6,0.001]
   print('----creating the signal, plz wait------')
   sig=gs.generate_signal(length_seconds, sampling_rate, freq_list, add_noise=True)
   print('finish : we start storing it in a csv file')
   path_='./data/{}'.format(name_)
   path__='./data/{}'.format(name__)

   if not os.path.exists(path_) :
      os.makedirs(path_)    

   gs.register_signal(sig[0],'./data/{}/signal'.format(name_))
     
   if not os.path.exists(path__):
      os.makedirs(path__)
   print('----we got it : time to create the ndarray-----')

   train_set,test_set,_=get_data(name_, batch_size=540, eval_batch_size=540, device=device)
   print('--------------- we got the data -------------')
   
   #Initiate an instance :
   ninp=1
   nhid=240
   nlayers=1
   nMLP=124
   nhead=12
   dropout=0.1
   epochs=1000

   # model=TransformerModel(ninp, nhead, nhid, nlayers, nMLP, bptt, pos_encod=True, dropout=dropout, device=device)
   # model.fit(train_set, test_set, epochs, name_)
   # test_loss = model.evaluate(test_set, val=True)
   # train_loss = model.evaluate(train_set, val=False, predict=True)
   # print('=' * 89)
   # print('| End of training | test loss {:5.2f} | train loss {:5.2f} | test ppl {:8.2f}'.format(test_loss, train_loss, math.exp(test_loss)))
   # print('=' * 89)
   # print('---------- Name of the file : {} --------------'.format(name_))
      
   model_=TransformerModel(ninp, nhead, nhid, nlayers, nMLP, bptt, pos_encod=False, dropout=dropout, device=device)
   model_.fit(train_set, test_set, epochs, name__)
   test_losss = model_.evaluate(test_set, val=True)
   train_losss = model_.evaluate(train_set, val=False, predict=True)
   print('=' * 89)
   print('| End of training | test loss {:5.2f} | train loss {:5.2f} | test ppl {:8.2f}'.format(test_losss, train_losss, math.exp(test_losss)))
   print('=' * 89)
   print('---------- Name of the file : {} --------------'.format(name__))

if __name__ == '__main__':
   parser=argparse.ArgumentParser()
   parser.add_argument('name', help='Name of the signal you will create')
   parser.add_argument('device', help='Processor used for torch tensor')
   parser.add_argument('-bptt', help='bptt :longueur des sequences etudiees')
   args=parser.parse_args()
   if not args.bptt==None :
      main(args.name,args.device, int(args.bptt))
   else :
      main(args.name, args.device)
