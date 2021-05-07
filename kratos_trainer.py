import sys
import torch
import torch.nn as nn
import time
import argparse
import generate_signal as gs
from load_data import *
from model import *
import os

def main(name,identifiant,device='cpu'):
    nlimit=0

    filename='nbeats_f100/train/SAT2_10_minutes_future100_{}.csv'.format(identifiant)
    filename_='nbeats_f100/test/SAT2_10_minutes_future100_4.csv'
   
    train_set=gs.register_signal(filename).transpose() #[time_step]x[dim] > [dim]x[time_step]
    test_set=gs.register_signal(filename_).transpose()

    train_set=normalize_data(train_set[nlimit:nlimit+1,:])
    test_set=normalize_data(test_set[nlimit:nlimit+1,:])

    path='./data/{}'.format(name)

    if not os.path.exists(path) :
        os.makedirs(path)
    np.savetxt('./data/{}/data_train_set.txt'.format(name), train_set)

    backast_length=60
    forecast_length=40
    nb=2500
    
    xtrain,ytrain,xtest,ytest=get_data2(backast_length, forecast_length, nb, train_set, test_set)
    print('we got the data : xtrain.shape :', xtrain.shape)
    
    #Initiate an instance :
    ninp=xtrain.shape[-1]
    nhid=256
    nlayers=2
    nMLP=128
    nhead=4
    dropout=0.2
    epochs=200
    bsz=256
    eval_bsz=256
    
    model=TransformerModel(ninp, nhead, nhid, nlayers, nMLP, backast_length, forecast_length, dropout=dropout, device=device)
    
    print("Model structure: ", model, "\n\n")
    for layer_name, param in model.named_parameters():
        print(f"Layer: {layer_name} | Size: {param.size()} \n")


    model.fit(xtrain, ytrain, xtest, ytest, bsz, eval_bsz, epochs, name)
    test_loss = model.evaluate(xtest, ytest, eval_bsz, True, name, predict=True)
    train_loss = model.evaluate(xtrain, ytrain, bsz, False, name, predict=True)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | train loss {:5.2f} | '.format(test_loss, train_loss))
    print('=' * 89)

    data_set=get_data_for_predict(backast_length, train_set)
    torch.save(data_set,'./data/{}/get_train_data_for_predict.pt'.format(name))
    model.evaluate_whole_signal(data_set,bsz,name)

    data_set_test=get_data_for_predict(backast_length, test_set)
    model.evaluate_whole_signal(data_set_test,eval_bsz,name,train=False)
        

    print('---------- Name of the file : {} --------------'.format(name))


    
if __name__ == '__main__':
   parser=argparse.ArgumentParser()
   parser.add_argument('name', help='The name of the folder in which out data will be saved')
   parser.add_argument('identifiant', help='Name of the signal you will create')
   parser.add_argument('-device', help='Processor used for torch')
   args=parser.parse_args()
   if args.device :
       main(args.name,int(args.identifiant), device=args.device)
   else :
       main(args.name,int(args.identifiant))
