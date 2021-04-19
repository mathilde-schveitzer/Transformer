import torch
import torch.nn as nn
import time
import argparse
import generate_signal as gs
from load_data import get_data2
from model import *
import os

def main(identifiant,test=False,device='cpu'):

   

    # we use the data store in the file

    filename='nbeats_f100/train/SAT2_10_minutes_future100_{}.csv'.format(identifiant)

    if test :
        filename_='nbeats_f100/test/SAT2_10_minutes_future100_4.csv'

    data_train=gs.register_signal(filename)

    print('-'*83 + ' D A T A TRAIN : ')
    print(data_train)

if __name__ == '__main__':
   parser=argparse.ArgumentParser()
   parser.add_argument('identifiant', help='Name of the signal you will create')
   parser.add_argument('-test', help='Set to true if you need the test set')
   parser.add_argument('-device', help='Processor used for torch tensor')
   args=parser.parse_args()
   if args.test :
       if args.device :
           main(int(args.identifiant), device=args.device)
       else :
           main(int(args,identifiant),test=bool(args.test))
   if args.test :
       main(int(args.identifiant),test=bool(args.test))
   else :
       main(int(args.identifiant))
