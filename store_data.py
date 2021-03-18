import sys
import os
sys.path.append(os.getcwd())
import argparse
import generate_signal as gs
from load_data import get_data
import numpy as np
import random as rd
import matplotlib.pyplot as plt

def main(name,device='cpu',samples=1500):
    
    ''' First programm to execute. It sets datas format, therefor the following parameters shouldn't be modified then :
    - backcast and forecast length 
    - iteration : determines the number of samples
    - signal : choose the caracteristics of the signal that will be analyzed.

    Datas are stored in file.txt easely exploitable, following the format : xtrain_name.txt'''
    #we create the directories that will be usefull afterward
    datapath='./data/{}/datas'.format(name)
    os.makedirs(datapath)
    os.makedirs('./data/{}/predictions'.format(name))
    os.makedirs('./data/{}/out'.format(name))
    # we generate the signal which will be analyzed
    length_seconds, sampling_rate=1000, 150 #that makes 15000 pts
    freq_list=[0.5]
    print('----creating the signal, plz wait------')
    sig=gs.generate_signal(length_seconds, sampling_rate, freq_list)
    print('finish : we start storing it in a csv file')
    gs.register_signal(sig[0],'./data/{}/signal'.format(name))
    print('----we got it : time to create the ndarray-----')
    
    xtrain,ytrain, xtest, ytest=get_data(backcast_length, forecast_length,limit, './data/{}/signal.csv'.format(name),copy=samples)
    np.savetxt(datapath+'/xtrain.txt',xtrain)
    np.savetxt(datapath+'/ytrain.txt',ytrain)
    np.savetxt(datapath+'/xtest.txt', xtest)
    np.savetxt(datapath+'/ytest.txt', ytest)
    print('--------- name of the file you used : {} ---------'.format(name))

   # trainandsave(name,device,nb)
    
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('name', help='Name of the storing file')
    parser.add_argument('-samples', help='Number of line your ndarray data will contain')
    parser.add_argument('-device', help='Processor used for torch tensor')
    args=parser.parse_args()
    if not args.samples :
        if not args.device :
            main(args.name)
        else :
            main(args.name,device=args.device)
    else :
        if not args.device :
            main(args.name,samples=int(args.samples))
        else :    
            main(args.name, args.device, int(args.samples))
