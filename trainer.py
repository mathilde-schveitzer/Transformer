import torch
import torch.nn as nn
import time
import argparse
import generate_signal as gs
from load_data import get_data2
from model import *

def main(name,device):

   #dimension of the input
   n=3
   ninp=3*n #due to generate_multi
   
   # we generate the signal which will be analyzed
   length_seconds, sampling_rate=1000, 150 #that makes 15000 pts
   freq_array=np.array([[0.05,0.1, 5],[1,2, 0.7],[0.3,3,2]])
   print('----creating the signal, plz wait------')
   sig=gs.generate_multi(n,length_seconds, sampling_rate, freq_array, add_noise=True)
   print('------- S I G N A L -------', sig.shape)
   print('finish : we start storing it in a txt file')
   path='./data/{}'.format(name)

   if not os.path.exists(path) :
      os.makedirs(path)    

   np.savetxt(sig,path+'txt')

   print('done')

   backast_length=10
   forecast_length=4 #chemin de la facilite

   nb_data=5000

   xtrain,ytrain,xtest,ytest=get_data2(backast_length, forecast_length, nb_data, sig, device=device)
   print('--------------- we got the data -------------')
   
   #Initiate an instance :
   nhid=256
   nlayers=1
   nMLP=128
   nhead=12
   dropout=0.1
   epochs=100
   bsz=128
   eval_bsz=128
   
   model=TransformerModel(ninp, nhead, nhid, nlayers, nMLP, backast_length, forecast_length, pos_encod=True, dropout=dropout, device=device)
   model.fit(xtrain, ytrain, xtest, ytest, bsz, eval_bsz, epochs, name, verbose=True)
   test_losss = model.evaluate(xtest, ytest, eval_bsz, val=True)
   train_losss = model.evaluate(xtrain, ytrain, bsz, val=False)
   print('=' * 89)
   print('| End of training | test loss {:5.2f} | train loss {:5.2f} | test ppl {:8.2f}'.format(test_losss, train_losss, math.exp(test_losss)))
   print('=' * 89)
   print('---------- Name of the file : {} --------------'.format(name))

if __name__ == '__main__':
   parser=argparse.ArgumentParser()
   parser.add_argument('name', help='Name of the signal you will create')
   parser.add_argument('device', help='Processor used for torch tensor')
   args=parser.parse_args()
   main(args.name,args.device)
