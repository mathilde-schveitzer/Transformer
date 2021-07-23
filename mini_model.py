import torch
import torch.nn as nn
import os


class TransformerModel(nn.Module):

    def __init__(self, backast_size, forecast_size, quantiles, ninp=1, nout=1, nhead=2, nhid=128, nlayers=1, dropout=0.2, device=torch.device('cpu')):
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
