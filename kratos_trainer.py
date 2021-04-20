import torch
import torch.nn as nn
import time
import argparse
import generate_signal as gs
from load_data import get_data2
from model import *
import os

def main(name,identifiant,device='cpu'):

   
    os.chdir('/data1/infantes/kratos/d2/nbeats_f100')

    # we use the data store in the file

    filename='./train/SAT2_10_minutes_future100_{}.csv'.format(identifiant)

    filename_='/test/SAT2_10_minutes_future100_4.csv'

    train_set=gs.register_signal(filename)
    test_set=gs.register_signal(filename_)

    path='./data/{}'.format(name)

    if not os.path.exists(path) :
        os.makedirs(path)    

    backast_length=10
    forecast_length=4 #chemin de la facilite

    xtrain,ytrain,xtest,ytest=get_data2(backast_length, forecast_length, nb, train_set, test_set, device=device)

if __name__ == '__main__':
   parser=argparse.ArgumentParser()
   parser.add_argument('name', help='The name of the folder in which out data will be saved')
   parser.add_argument('identifiant', help='Name of the signal you will create')
   parser.add_argument('-device', help='Processor used for torch')
   args=parser.parse_args()
   if args.device :
       main(int(args.identifiant), device=args.device)
   else :
       main(int(args.identifiant))
