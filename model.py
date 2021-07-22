import os
import numpy as np
import math
import torch
import time
import random
import torch.nn as nn
import torch.nn.functional as F
from load_data import *

# ---- Definition du modele et des methodes permettant de l'entrainer

class TransformerModel(nn.Module):

    def __init__(self, ninp, nhead, nhid, nlayers, backast_size, forecast_size, dropout=0.5, device=torch.device('cpu')):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.embed_dims=ninp*nhead
        self.device = device
        encoder_layers = TransformerEncoderLayer(self.embed_dims, nhead, nhid, dropout, activation='gelu')

        self.encoder=nn.Linear(ninp, self.embed_dims)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)       
        self.decoder=Decoder(self.embed_dims, ninp, forecast_size,backast_size,device=device)


        self.parameters = []
        self.parameters = nn.ParameterList(self.parameters)
        
        self.optimizer=torch.optim.Adam(self.parameters(),lr=1e-3,betas=(0.9,0.98))
        self._loss=F.l1_loss

        self.to(self.device)
        
        print('|T R A N S F O R M E R : Optimus Prime is ready |')

    def forward(self, input):
        input=input.to(self.device)        
        input = self.encoder(input)
        output = self.transformer_encoder(input)
        _output = self.decoder(output)
        return _output

# main function, training the model

    def fit(self, x_train, y_train, x_test, y_test, batch_size, epochs, filename, save=True, verbose=False):
        store_test_loss=np.zeros(epochs)
        store_loss=np.zeros(epochs)
        time_=np.zeros(epochs)
        
    # first iteration is done separatedly    
        test_loss= self.evaluate(x_test, y_test, batch_size, filename, False)
        train_loss=self.evaluate(x_train, y_train, batch_size, filename, True)
        store_loss[0]=train_loss
        store_test_loss[0]=test_loss
        
        print('test loss--------- : {}'.format(test_loss))
        print('train loss-------- : {}'.format(train_loss))
        np.savetxt('./data/{}/train_loss.txt'.format(filename), store_loss)
        np.savetxt('./data/{}/val_loss.txt'.format(filename), store_test_loss)

        t0=time.time()
        time_[0]=t0
        
        for epoch in range(1, epochs):
            
        # shuffle before spliting in batch to avoid overfitting
            x_train, y_train=shuffle_in_unison(x_train,y_train)
            x_train_list = split(x_train, batch_size)
            y_train_list = split(y_train, batch_size)
            
        # to keep an eye on what is happening
            log_epoch=1
            if epoch % log_epoch == 0 :
                print('|---------------- Epoch noumero {} ----------|'.format(epoch))

            epoch_start_time=time.time()
            self.do_training(x_train_list, y_train_list)
            elapsed_time=time.time()-epoch_start_time

        # compute and store the loss
            test_loss= self.evaluate(x_test, y_test, batch_size, filename, False)
            train_loss=self.evaluate(x_train, y_train, batch_size, filename, True)
            store_loss[epoch]=train_loss
            store_test_loss[epoch]=test_loss
            print('test loss--------- : {}'.format(test_loss))
            print('train loss-------- : {}'.format(train_loss))
            np.savetxt('./data/{}/train_loss.txt'.format(filename), store_loss)
            np.savetxt('./data/{}/val_loss.txt'.format(filename), store_test_loss)
            time_[epochs]=time.time()-t0

# subsidiary function, called by fit
    def do_training(self,xtrain,ytrain):
        self.train() # Turn on the train mode (herited from module)
        total_loss = 0.
        start_time = time.time()

        assert len(xtrain)==len(ytrain)
        
        for batch_id in range(len(xtrain)):

        # creating objects that can be used by the model
            data, targets = torch.tensor(xtrain[batch_id],dtype=torch.float).to(self.device), torch.tensor(ytrain[batch_id], dtype=torch.float).to(self.device)
            data=data.transpose(0,1)

        # old fashion
            self.optimizer.zero_grad()
            output = self(data)            
            loss=self._loss(output.reshape(-1), targets.transpose(0,1).reshape(-1).to(self.device))            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.optimizer.step()

            total_loss += loss.item()

            log_interval = 50
            if batch_id % log_interval == 0 :
                if batch_id==0 :
                    cur_loss=total_loss
                else :
                    cur_loss = total_loss / log_interval
                print(' {:5d}/{:5d} batches | ''loss {:5.2f}' .format(batch_id, len(xtrain), cur_loss))
        
                total_loss=0
# function called to compute the loss and use the model without training it
    def evaluate(self, xtest, ytest, bsz, val, name, predict=False, verbose=False):  
        self.eval() # Turn on the evaluation mode (herited from module)

        xtest_list=split(xtest, bsz)
        ytest_list=split(ytest, bsz)
        assert len(xtest_list)==len(ytest_list)

    # to store prediction   
        if predict :
            prediction=torch.zeros_like(ytest)
            
        test_loss = []
        
        for batch_id in range(0, len(xtest_list)):
       
            data,targets=torch.tensor(xtest_list[batch_id], dtype=torch.float).to(self.device), torch.tensor(ytest_list[batch_id], dtype=torch.float).to(self.device)
            data=data.transpose(0,1)
            output=self(data)
            loss=self._loss(output.reshape(-1),targets.transpose(0,1).reshape(-1).to(self.device)).item()
            test_loss.append(loss)

            if predict :
                #d'ou l'importance de passer les batch dans l'ordre
                output=output.transpose(0,1)
                prediction[batch_id*output.shape[0]:(batch_id+1)*output.shape[0],:,:]=output
                print(prediction)
                
        mean_loss=np.mean(test_loss)

        if val :
            print(f'Validation loss : {mean_loss:.4f}')
        else :
            print(f'Training loss : {mean_loss :.4f}')
            
        if predict :
            if val :
                print('-------we save test values----------')
                torch.save(prediction,'./data/{}/predictions_test.pt'.format(name))
            else :
                print('-------we save train values ------')
                torch.save(prediction, './data/{}/predictions_train.pt'.format(name))
        return(mean_loss)



# ------ definition des sous classes utilisees dans la definition de TransformerModel
class MLForecast(nn.Module) :
    def __init__(self, forecast_size, backast_size, device='cpu') :
        super(MLForecast, self).__init__()
        self.MLP=nn.Linear(backast_size, forecast_size)
        self.to(device)

    def forward(self, x):
        input=x.transpose(0,2) # "[forecast_size][batch][ninp] => [ninp][batch][forecast_size]"
        output=self.MLP(x)
        return(output)

class Decoder(nn.Module) :
    def __init__(self, ninp, nout, forecast_size, backast_size, device='cpu') :
       super(Decoder, self).__init__()
       self.MLP=nn.Linear(ninp, nout)
       self.MLF=nn.Linear(backast_size,forecast_size)
       self.to(device)

    def forward(self,x,verbose=False):
        x=self.MLP(x)
        if verbose :
            print('encoder input :', x.shape)
        output=self.MLF(x.transpose(0,2))
        if verbose :
            print('encoder output :', output.shape)            
        return(output.transpose(0,2))
