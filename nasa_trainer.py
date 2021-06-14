import sys
import torch
import argparse
from load_data import *
from nbeats import *
import os


def main(name,id):

    filename='NASA/train/{}.csv'.format(id)
    filename_='NASA/test/{}.csv'.format(id)
    
    data_set=read_nasa_signal(filename)
    data_set_=read_nasa_signal(filename_)

    path='./data/{}'.format(name)
    if not os.path.exists(path) :
        os.makedirs(path)

    np.savetxt('./data/{}/data_train.txt'.format(name), data_set) #in order to plot it easily
    print(data_set.shape)
    print(data_set_.shape)
    data_set=np.vstack((data_set,data_set_))
    
    data_set=normalize_datas(data_set.transpose())  #[time_step]x[dim] > [dim]x[time_step]
                       
    backcast_length=100
    forecast_length=100
    step=10

    x,y = get_data(backcast_length, forecast_length, step, data_set) # x=[Nechantillon][b_l][Ndim]

    xtrain,ytrain,xtest,ytest=split_set(x,y)
    
    print('we got the data : xtrain.shape :', xtrain.shape)
    print(ytrain.shape)
    print(xtest.shape)
    print(ytest.shape)


    
    

    
    torch.save(xtrain,'./data/{}/xtrain.pt'.format(name))
    torch.save(ytrain,'./data/{}/ytrain.pt'.format(name))
    torch.save(ytest,'./data/{}/ytest.pt'.format(name))
    torch.save(xtest,'./data/{}/xtest.pt'.format(name))

    print('---------- Name of the file : {} --------------'.format(name))


    
if __name__ == '__main__':
   parser=argparse.ArgumentParser()
   parser.add_argument('name', help='The name of the folder in which out data will be saved')
   parser.add_argument('id', help='the few strings that enable to identify the file you want to analyze')
   args=parser.parse_args()
   main(args.name, args.id)
