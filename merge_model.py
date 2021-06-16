from nbeats import NBeatsNet
import os
import numpy as np
import math
import torch
import time
import random
import torch.nn as nn
import torch.nn.functional as F
from load_data import *
from torch.nn import TransformerDecoder, TransformerDecoderLayer

class MergedModel(nn.Module):

    def __init__(self, ninp,
                 nhead=2,
                 backcast_length=100,
                 forecast_length=100,
                 device=torch.device('cpu')):

        super(MergedModel, self).__init__()


        self.encoder = NBeatsNet(ninp, forecast_length=forecast_length, backcast_length=backcast_length, device=device)
        embed_dims=ninp*nhead
        decoder_layers = TransformerDecoderLayer(embed_dims, nhead)
        self.converter = nn.Linear(ninp, embed_dims)
        self.decoder = TransformerDecoder(decoder_layers, num_layers=4)
        self.deconverter = nn.Linear(embed_dims, ninp)
        self.device=device

        
        self._opt = torch.optim.Adam(self.parameters(), lr=1e-3, amsgrad=True)
        self._loss = F.l1_loss
        self.to(self.device)

        print('| R E A D Y  T O  C O M P U T E |')
        
        

    def forward(self, input) :
        _,memory = self.encoder(input) # [bsz, forecast_length, ninp]
        memory=self.converter(memory).transpose(0,1)
        tgt=torch.empty_like(memory)
        output=self.deconverter(self.decoder(tgt,memory))
        
        return(output.transpose(0,1))

    def do_training(self,xtrain,ytrain):
        self.train() # Turn on the train mode (herited from module)
        total_loss = 0.
        start_time = time.time()

        assert len(xtrain)==len(ytrain)
        
        for batch_id in range(len(xtrain)):
           
            data, targets = torch.tensor(xtrain[batch_id],dtype=torch.float).to(self.device), torch.tensor(ytrain[batch_id], dtype=torch.float).to(self.device)
            
            self._opt.zero_grad()
            output = self(data)            
            loss=self._loss(output.reshape(-1), targets.transpose(0,1).reshape(-1).to(self.device))
                        
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self._opt.step()

            total_loss += loss.item()

            log_interval = 50
            if batch_id % log_interval == 0 :
                if batch_id==0 :
                    cur_loss=total_loss
                else :
                    cur_loss = total_loss / log_interval
                print(' {:5d}/{:5d} batches | ''loss {:5.2f}' .format(batch_id, len(xtrain), cur_loss))
        
                total_loss=0

    def evaluate(self, xtest, ytest, bsz, val, name, predict=False):  
        self.eval() # Turn on the evaluation mode (herited from module)

        xtest_list=split(xtest, bsz)
        ytest_list=split(ytest, bsz)
        assert len(xtest_list)==len(ytest_list)
       
        if predict :
            prediction=torch.zeros_like(ytest)
            
        test_loss = []
        
        for batch_id in range(0, len(xtest_list)):
       
            data,targets=torch.tensor(xtest_list[batch_id], dtype=torch.float).to(self.device), torch.tensor(ytest_list[batch_id], dtype=torch.float).to(self.device)
            # data.shape = [bsz, backast_length, ninp]
            output=self(data)
            loss=self._loss(output.reshape(-1),targets.reshape(-1).to(self.device)).item()
            test_loss.append(loss)

            if predict :
                #d'ou l'importance de passer les batch dans l'ordre
                output=output.transpose(0,1)
                prediction[batch_id*output.shape[0]:(batch_id+1)*output.shape[0],:,:]=output
                print(prediction)
                
        mean_loss=np.mean(test_loss)
            
        if predict :
            if val :
                print('-------we save test values----------')
                torch.save(prediction,'./data/{}/predictions_test.pt'.format(name))
            else :
                print('-------we save train values ------')
                torch.save(prediction, './data/{}/predictions_train.pt'.format(name))
        return(mean_loss)


    def fit(self, x_train, y_train, x_test, y_test, filename, epochs, batch_size):
        store_test_loss=np.zeros(epochs)
        store_loss=np.zeros(epochs)

        x_test_list=split(x_test, batch_size)
        y_test_list =split(y_test, batch_size)

        test_loss= self.evaluate(x_test, y_test, batch_size, True, filename)
        train_loss=self.evaluate(x_train, y_train, batch_size, False, filename)
        store_loss[0]=train_loss
        store_test_loss[0]=test_loss
        time_=np.zeros(epochs)
        print('test loss--------- : {}'.format(test_loss))
        print('train loss-------- : {}'.format(train_loss))
        np.savetxt('./data/{}/train_loss.txt'.format(filename), store_loss)
        np.savetxt('./data/{}/val_loss.txt'.format(filename), store_test_loss)
        t0=time.time()
            
        for epoch in range(1, epochs):
            x_train, y_train=shuffle_in_unison(x_train,y_train)
            x_train_list = split(x_train, batch_size)
            y_train_list = split(y_train, batch_size)
            log_epoch=1
            if epoch % log_epoch == 0 :
                print('|---------------- Epoch noumero {} ----------|'.format(epoch))

            epoch_start_time=time.time()
            self.do_training(x_train_list, y_train_list)
            elapsed_time=time.time()-epoch_start_time
            
            test_loss= self.evaluate(x_test, y_test, batch_size, filename, False)
            train_loss=self.evaluate(x_train, y_train, batch_size, filename, True)
            store_loss[epoch]=train_loss
            store_test_loss[epoch]=test_loss
            print('test loss--------- : {}'.format(test_loss))
            print('train loss-------- : {}'.format(train_loss))
            np.savetxt('./data/{}/train_loss.txt'.format(filename), store_loss)
            np.savetxt('./data/{}/val_loss.txt'.format(filename), store_test_loss)
            time_[epoch]=time.time()-t0

        
