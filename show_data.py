import numpy as np
import matplotlib.pyplot as plt
import argparse


def main(signal, datatype):

    fig=plt.figure(figsize=(10,10))
    if datatype=='loss':
        train_pass='./data/{}/train_loss.txt'.format(signal)
        val_pass='./data/{}/validation_loss.txt'.format(signal)
        numpy_train=np.loadtxt(train_pass)
        numpy_val=np.loadtxt(val_pass)
        plt.plot(numpy_train,label= 'Erreur sur le training test')
        plt.plot(numpy_val, label='Erreur sur le validation test')
    plt.legend()
    plt.savefig(fig)

if __name__ == '__main__' :

    parser=argparse.ArgumentParser()
    parser.add_argument('name', help='Name of the signal analyzed')
    parser.add_argument('data_type', help='Precise what you want to show (options : loss)')

    args=parser.parse_args()
    main(args.name,args.data_type)
    
