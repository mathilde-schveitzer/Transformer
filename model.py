import os
import numpy as np
import math
import torch
import time
import random
import torch.nn as nn
import torch.nn.functional as F
import load_data as ld

class TransformerModel(nn.Module):

    def __init__(self, ninp, nhead, nhid, nlayers, nMLP, pos_encod=False, learning_rate=1e-5, dropout=0.5, device=torch.device('cpu')):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.embed_dims=ninp*nhead
        self.device = device
        self.ninp=ninp
        encoder_layers = TransformerEncoderLayer(self.embed_dims, nhead, nhid, dropout, activation='gelu')
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder=MLP(ninp,self.embed_dims,1, nMLP,device=self.device)
        self.pos_encod=pos_encod
        if pos_encod :
            self.pos_encoder=PositionalEncoding(self.embed_dims,dropout)
        self.decoder=MLP(self.embed_dims,ninp,1, nMLP,device=self.device)
        self.to(self.device)
        self.parameters = []
        self.parameters = nn.ParameterList(self.parameters)
        self.optimizer=torch.optim.Adam(lr=1e-2, params=self.parameters())
        self._loss=F.l1_loss
        print('|T R A N S F O R M E R : Optimus Prime is ready |')

    def forward(self, input, verbose=False):
        input = self.encoder(input).to(self.device)
       
        if self.pos_encod :
            if verbose :
                input=self.pos_encoder(input, verbose=True)
            else :
                input=self.pos_encoder(input)

        if verbose :
            print('----------pos_encoder : ', input.shape)

        output = self.transformer_encoder(input).to(self.device)

        if verbose :
            print('---------- transformer : ', output.shape)

        _output = self.decoder(output).to(self.device)

        if verbose :
            print('---------- decoder : ',_output.shape)

        return _output
    
    def do_training(self,xtrain,ytrain,verbose):
        self.train() # Turn on the train mode (herited from module)
        total_loss = 0.
        start_time = time.time()

        assert len(xtrain)==len(ytrain)

        shuffled_indices=list(range(len(xtrain)))
        random.shuffle(shuffled_indices)
        
        for batch_id in shuffled_indices:
           
            data, targets = xtrain[batch_id],ytrain[batch_id]
            data=data.transpose(0,1).to(self.device)
            targets=targets.transpose(0,1).to(self.device)

            if verbose :
                print('--------data :', data.shape)
                print('--------target :', targets.shape)
           
            self.optimizer.zero_grad()
            output = self(data)
            # output=ld.squeeze_last_dim(output).reshape(-1)
            loss=self._loss(output, targets)
         
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.optimizer.step()

            total_loss += loss.item()
            log_interval = 1
            if batch_id % log_interval == 0 :
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print(' {:5d}/{:5d} batches | ms/batch {:5.2f} | ''loss {:5.2f}' .format(batch_id, len(xtrain),elapsed * 1000 / log_interval,cur_loss))
        
                start_time = time.time()
            total_loss=0

    def evaluate(self,xtest, ytest, bsz, val, verbose=False):  
        self.eval() # Turn on the evaluation mode (herited from module)

        def split(arr, size):
           arrays = []
           while len(arr) > size:
               slice_ = arr[:size]
               arrays.append(slice_)
               arr = arr[size:]
           arrays.append(arr)
           return arrays

        xtest_list=split(xtest, bsz)
        ytest_list=split(ytest,bsz)
        
        assert len(xtest_list)==len(ytest_list)

        test_loss = []

        if verbose :
            if val :
                print('---VAL---')
        
            else :
                print('---PAVAL---')
        
        for batch_id in range(0, len(xtest_list)):
       
            data,targets=xtest_list[batch_id],ytest_list[batch_id]
            data=data.transpose(0,1).to(self.device)
            targets=targets.transpose(0,1).to(self.device)
            output=self(data)
            loss=self._loss(output,targets.to(self.device)).item()
            test_loss.append(loss)

        mean_loss=np.mean(test_loss)

        if val :
            print(f'Validation loss : {mean_loss:.4f}')
        else :
            print(f'Training loss : {mean_loss :.4f}')
      
        return(mean_loss)


    def fit(self, xtrain, ytrain, xtest, ytest, bsz, eval_bsz, epochs, filename, verbose=False):
        store_val_loss=np.zeros(epochs)
        store_loss=np.zeros(epochs)

        def split(arr, size):
            arrays = []
            while len(arr) > size:
                slice_ = arr[:size]
                arrays.append(slice_)
                arr = arr[size:]
            arrays.append(arr)
            return arrays

      
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            xtrain_list=split(xtrain, bsz)
            ytrain_list=split(ytrain, bsz)
            
            self.do_training(xtrain_list,ytrain_list,False)
           
            val_loss = self.evaluate(xtest, ytest, eval_bsz, val=True, verbose=True)
            train_loss = self.evaluate(xtrain, ytrain, bsz, val=False, verbose=False)
            store_loss[epoch-1]=train_loss
            store_val_loss[epoch-1]=val_loss
            
       
        np.savetxt('./data/{}/train_loss.txt'.format(filename), store_loss)
        np.savetxt('./data/{}/val_loss.txt'.format(filename), store_val_loss)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x, verbose=False):
        pencod=self.pe[:x.size(0), :]
        if verbose :
            print('-----------------------x ------------------------')
            print(x.shape)
            print('--------------------------- pencod ---------------------')
            print(pencod)
            print(pencod.shape)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MLP(nn.Module):
    def __init__(self, input_size, output_size, nhidden, dim_hidd, device):
        super(MLP, self).__init__()
        self.input_size=input_size
        self.output_size=output_size
        self.fc_layers=[]
        self.fc_layers.append(nn.Linear(self.input_size, dim_hidd).to(device))
        for k in range(nhidden):
            self.fc_layers.append(nn.Linear(dim_hidd, dim_hidd).to(device))
        self.fc_layers.append(nn.Linear(dim_hidd, self.output_size).to(device))
        self.device=device
        self.to(device)
        
    def forward(self,x):
        output=x
        for f in self.fc_layers :
            output=f(output)
        return(output) 
