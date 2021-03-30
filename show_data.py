import numpy as np
import matplotlib.pyplot as plt
import argparse


def main(signal, datatype):

    fig=plt.figure(figsize=(10,10))

    if datatype=='loss':
        train_pass='./data/{}/train_loss.txt'.format(signal)
        val_pass='./data/{}/val_loss.txt'.format(signal)
        numpy_train=np.loadtxt(train_pass)
        numpy_val=np.loadtxt(val_pass)
        plt.plot(numpy_train,label= 'Erreur sur le training test')
        plt.plot(numpy_val, label='Erreur sur le validation test')
   

    # if datatype=='pos_encod':
    #     name=signal+'_encod'
    #     name_=signal+'_no_encod'
        
    #     train_pass='./data/{}/train_loss.txt'.format(name)
    #     val_pass='./data/{}/val_loss.txt'.format(name)
    #     numpy_train=np.loadtxt(train_pass)
    #     numpy_val=np.loadtxt(val_pass)
    #     plt.plot(numpy_train,label= 'Erreur sur le training test - with pos encoding')
    #     plt.plot(numpy_val, label='Erreur sur le validation test - with pos encoding')

    #     train_pass2='./data/{}/train_loss.txt'.format(name_)
    #     val_pass2='./data/{}/val_loss.txt'.format(name_)
    #     numpy_train_=np.loadtxt(train_pass2)
    #     numpy_val_=np.loadtxt(val_pass2)
    #     plt.plot(numpy_train_,label= 'Erreur sur le training test - without  pos encoding')
    #     plt.plot(numpy_val_, label='Erreur sur le validation test - without pos encoding')
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

    plt.legend()
    plt.savefig('./data/{}/loss')
        
if __name__ == '__main__' :

    parser=argparse.ArgumentParser()
    parser.add_argument('name', help='Name of the signal analyzed')
    parser.add_argument('data_type', help='Precise what you want to show (options : loss)')

    args=parser.parse_args()
    main(args.name,args.data_type)
    
