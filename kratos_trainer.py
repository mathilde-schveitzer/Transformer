import sys
import torch
import torch.nn as nn
import time
import argparse
from load_data import *
from nbeats import *
import os

def main(name,nlimit,device='cpu'):
    
    test_path='nbeats_f100/test/SAT2_10_minutes_future100_4.csv'
   
    train_set=register_training_signal(nlimit).transpose() #[time_step]x[dim] > [dim]x[time_step]
    test_set=read_signal(test_path).transpose()
    test_set=test_set[nlimit:nlimit+1,:]
    train_set=np.expand_dims(train_set[-1,:],0) # comment if you want the [0,nlimit] signals

    # if test_set.shape[0]==1 :
    #      test_set=normalize_data(test_set)
    #      train_set=normalize_data(train_set)

    # else :
    #     test_set=normalize_datas(test_set)
    #     train_set=normalize_datas(train_set)
        
    print(test_set.shape)
    print(train_set.shape)


    
    path='./data/{}'.format(name)

    if not os.path.exists(path) :
        os.makedirs(path)
    np.savetxt('./data/{}/data_train_set.txt'.format(name), train_set)
    np.savetxt('./data/{}/data_test_set.txt'.format(name), test_set)
    

    backast_length=80
    forecast_length=80
    ninterval=backast_length//10
    
    xtrain,ytrain,xtest,ytest=get_data2(backast_length, forecast_length, ninterval, train_set, test_set)
    print('we got the data : xtrain.shape :', xtrain.shape)
    torch.save(xtrain,'./data/{}/xtrain.pt'.format(name))
    torch.save(ytrain,'./data/{}/ytrain.pt'.format(name))
    torch.save(ytest,'./data/{}/ytest.pt'.format(name))
    torch.save(xtest,'./data/{}/xtest.pt'.format(name))


    
    #Initiate an instance :
    
    epochs=1000
    bsz=128
    eval_bsz=128
   
    model=NBeatsNet(device=device, forecast_length=forecast_length, backcast_length=backast_length)
    
    print("Model structure: ", model, "\n\n")
    for layer_name, param in model.named_parameters():
        print(f"Layer: {layer_name} | Size: {param.size()} \n")

    start_time=time.time()
    model.fit(xtrain, ytrain, xtest, ytest, name, epochs=epochs, batch_size=bsz)
    elapsed_time=time.time()-start_time
    test_loss = model.evaluate(xtest, ytest, eval_bsz, name, False, save=True)
    train_loss = model.evaluate(xtrain, ytrain, bsz, name, True, save=True)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | train loss {:5.2f} | '.format(test_loss, train_loss))
    print('=' * 89)
    print('| DL Session took {} seconds |'.format(elapsed_time))

    # Not working on this branch
    # data_set=get_data_for_predict(backast_length, train_set)
    # torch.save(data_set,'./data/{}/get_train_data_for_predict.pt'.format(name))
    # model.evaluate_whole_signal(data_set,bsz,name)

    # data_test_set=get_data_for_predict(backast_length, test_set)
    # torch.save(data_test_set, './data/{}/get_test_data_for_predict.pt'.format(name))
    # model.evaluate_whole_signal(data_test_set,eval_bsz,name,train=False)
        

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
