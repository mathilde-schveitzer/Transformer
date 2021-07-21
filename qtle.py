import os
import numpy as np
import torch
import torch.nn as nn
from load_data import *


#----- Definition of the model

class TransformerModel(nn.Module):

    def __init__(self, backast_size, forecast_size, quantiles, ninp=1, nout=1, nhead=2, nhid=128, nlayers=2, dropout=0.2, device=torch.device('cpu')):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.embed_dims=ninp*nhead
        self.device = device
        encoder_layers = TransformerEncoderLayer(self.embed_dims, nhead, nhid, dropout, activation='gelu')
        self.encoder=nn.Linear(ninp, self.embed_dims)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.decoders=[]
        for _ in quantiles :
            decoder=Decoder(self.embed_dims, ninp, backast_size, forecast_size)
            self.decoders.append(decoder)
        self.parameters = []
        self.parameters = nn.ParameterList(self.parameters)
        
        self.to(self.device)
        
        print('|T R A N S F O R M E R : Optimus Prime is ready |')

    def forward(self, input, idx):
        input=input.to(self.device)    
        input = self.encoder(input)
        output = self.transformer_encoder(input)
        # last output depends of the quantile
        _output = self.decoders[idx](output)
        return _output

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

# Define the loss that would be used to estimate the quantiles

class QuantileLoss(torch.nn.Module):
    def __init__(self, tau):
        super(QuantileLoss, self).__init__()
        self.tau=tau
        
    def forward(self, yhat, y):
        diff = yhat - y
        mask = (diff.ge(0).float() - self.tau).detach() #-tau or 1-tau
        return (mask * diff).mean()

# Define the quantiles we'd like to estimate

quantiles = [0.05, 0.5, 0.95]

# Create a model and fix hyperparameters 

bks=100
fks=100
batch_size=50
step=10
device='cpu'
model=TransformerModel(bks,fks,quantiles)
optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(0.9,0.98))
name='test1'

# get the data

data_set=register_training_signal(nlimit=0).transpose() #[time_step]x[dim] > [dim]x[time_step]
data_set=normalize_datas(data_set)
x_train, y_train = get_data(bks, fks, step, data_set)
print('|------------------ xtrain.shape : {} | ', x_train.shape)
print('|------------------ ytrain.shape : {} | ', y_train.shape)

# create a storage directory

path='./data/{}'.format(name)
if not os.path.exists(path) :
    os.makedirs(path)

# Loss for each session :
losses=[QuantileLoss(tau) for tau in quantiles]

epochs=50
log_epoch=1
store_loss=np.zeros(epochs)

print('| R E A D Y  T O  T R A I N : VROUM  VROUM |')

for epoch in range(epochs) :

    # Split data to batches, shuffle to avoid overfitting
    x_train, y_train=shuffle_in_unison(x_train,y_train)
    x_train_list = split(x_train, batch_size)
    y_train_list = split(y_train, batch_size)
    total_loss=0
    loss_to_store=0

    for batch_id in range(len(x_train_list)):
        #convert data to expected format
        data, targets = torch.tensor(x_train_list[batch_id],dtype=torch.float).to(device), torch.tensor(y_train_list[batch_id], dtype=torch.float).to(device)
        data, targets = data.transpose(0,1), targets.transpose(0,1)
        optimizer.zero_grad()
        err=torch.zeros(len(quantiles))
        for idx,tau in enumerate(quantiles) :
            output = model(data,idx)
            loss=losses[idx](output.reshape(-1), targets)
            err[idx]=loss
        total_loss=torch.mean(err)
        total_loss.backward()
        optimizer.step()
        loss_to_store+=total_loss.item()

# store loss and print the result during training session

    store_loss[epoch]=loss_to_store
    np.savetxt('./data/{}/train_loss.txt'.format(name), store_loss)

    if epoch % log_epoch == 0 :
        print('|---------------- Epoch noumero {} ----------|'.format(epoch))
        print('|---------------- Loss : {} -----------------|'.format(loss_to_store))
        
# TODO : save the model so obtained

