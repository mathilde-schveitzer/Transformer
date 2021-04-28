import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import random
from load_data import merge_data

def main(signal, datatype,show=True):

    fig=plt.figure()

    if datatype=='loss':
        train_pass='./data/{}/train_loss.txt'.format(signal)
        val_pass='./data/{}/val_loss.txt'.format(signal)
        numpy_train=np.loadtxt(train_pass)
        numpy_val=np.loadtxt(val_pass)
        plt.plot(numpy_train,label= 'Erreur sur le training test')
        plt.plot(numpy_val, label='Erreur sur le validation test')

    if datatype=='pos_encod':
        name=signal+'_encod'
        name_=signal+'2'
        
        train_pass='./data/{}/train_loss.txt'.format(name)
        val_pass='./data/{}/val_loss.txt'.format(name)
        numpy_train=np.loadtxt(train_pass)
        numpy_val=np.loadtxt(val_pass)
        plt.plot(numpy_train,label= 'Erreur sur le training test - with pos encoding')
        plt.plot(numpy_val, label='Erreur sur le validation test - with pos encoding')
        
        train_pass2='./data/{}/train_loss.txt'.format(name_)
        val_pass2='./data/{}/val_loss.txt'.format(name_)
        numpy_train_=np.loadtxt(train_pass2)
        numpy_val_=np.loadtxt(val_pass2)
        plt.plot(numpy_train_,label= 'Erreur sur le training test - without  pos encoding')
        plt.plot(numpy_val_, label='Erreur sur le validation test - without pos encoding')

    if datatype=='prediction':
        predict_path='./data/{}/predictions_train.pt'.format(signal)
        xtrain_path='./data/{}/xtrain.pt'.format(signal)
        ytrain_path='./data/{}/ytrain.pt'.format(signal)
        
        
        prediction=torch.load(predict_path,map_location=torch.device('cpu')) # [nb]x[forecast_size]x[dim]
        data=torch.load(xtrain_path, map_location=torch.device('cpu'))
        target=torch.load(ytrain_path, map_location=torch.device('cpu'))

        print(prediction.shape)
        print(data.shape)
        prediction=prediction[:,:,0].detach().numpy()
        data=data[:,:,0].numpy()
        target=target[:,:,0].numpy()

        def merge_line(a,b,k):
            merge=np.zeros(a.shape[1]+b.shape[1])
            merge[:a.shape[1]]=a[k,:]
            merge[a.shape[1]:]=b[k,:]
            return(merge)

        k=random.randrange(prediction.shape[0])
        plt.plot(merge_line(data,target,k),label='target')
        plt.plot(merge_line(data,prediction,k),label='prediction')
                
    if datatype=='whole_prediction':
        predict_path='./data/{}/predictions_train.pt'.format(signal)
        xtrain_path='./data/{}/xtrain.pt'.format(signal)
        ytrain_path='./data/{}/ytrain.pt'.format(signal)
        
        
        
        prediction=torch.load(predict_path,map_location=torch.device('cpu')) # [nb]x[forecast_size]x[dim]
        data=torch.load(xtrain_path, map_location=torch.device('cpu'))
        target=torch.load(ytrain_path, map_location=torch.device('cpu'))

        data=data[:,:,0].detach().numpy()
        target=target[:,:,0].detach().numpy()
        prediction=prediction[:,:,0].detach().numpy()

        merge_target=merge_data(data,target)
        merge_prediction=merge_data(data,prediction)
        plt.plot(merge_target, label='target')
        plt.plot(merge_prediction, label='prediction')


    plt.legend()
    plt.show()
x
        
if __name__ == '__main__' :

    parser=argparse.ArgumentParser()
    parser.add_argument('name', help='Name of the signal analyzed')
    parser.add_argument('data_type', help='Precise what you want to show (options : loss)')
    parser.add_argument('-show', help='Type n if you do no want to show the picture (only saved)')
    
    args=parser.parse_args()
    if args.show=='n' :
        main(args.name, args.data_type, show=False)
    else :
        main(args.name, args.data_type)
    
    
