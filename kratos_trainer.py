import sys
import torch
import argparse
from load_data import *
from nbeats import *
import os


def main(name,nlimit):
    
    data_set=register_training_signal(nlimit).transpose() #[time_step]x[dim] > [dim]x[time_step]

   # data_set=np.expand_dims(data_set[-1,:],0) # comment if you want the [0,nlimit] signals

    if data_set.shape[0]==1 :
        data_set=normalize_data(data_set)
    else :
        data_set=normalize_datas(data_set)

    print(data_set.shape)
    
    backcast_length=200
    forecast_length=200
    step=10

    x,y = get_data(backcast_length, forecast_length, step, data_set) # x=[Nechantillon][b_l][Ndim]

    xtrain,ytrain,xtest,ytest=split_set(x,y)
    
    print('we got the data : xtrain.shape :', xtrain.shape)
    print(ytrain.shape)
    print(xtest.shape)
    print(ytest.shape)

    path='./data/{}'.format(name)
    
    if not os.path.exists(path) :
        os.makedirs(path)

    
    torch.save(xtrain,'./data/{}/xtrain.pt'.format(name))
    torch.save(ytrain,'./data/{}/ytrain.pt'.format(name))
    torch.save(ytest,'./data/{}/ytest.pt'.format(name))

    torch.save(xtest,'./data/{}/xtest.pt'.format(name))
    print('---------- Name of the file : {} --------------'.format(name))


    
if __name__ == '__main__':
   parser=argparse.ArgumentParser()
   parser.add_argument('name', help='The name of the folder in which out data will be saved')
   parser.add_argument('nlimit', help='number of signals which will be stored')
   args=parser.parse_args()
   main(args.name, int(args.nlimit))
