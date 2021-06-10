import sys
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from load_data import split, shuffle_in_unison
from model import TransformerModel

class NBeatsNet(nn.Module):
    SEASONALITY_BLOCK = 'seasonality'
    TREND_BLOCK = 'trend'
    GENERIC_BLOCK = 'generic'

    def __init__(self,
                 ninp,
                 device=torch.device('cpu'),
                 block_types=(GENERIC_BLOCK,GENERIC_BLOCK),
                 forecast_length=5,
                 backcast_length=10,
                 thetas_dim=(8,8),
                 hidden_layer_units=8,
                 block_type='fully_connected',
                 nb_harmonics=None):
        
        super(NBeatsNet, self).__init__()

        self.ninp=ninp
        self.forecast_length = forecast_length
        self.backcast_length = backcast_length
        self.hidden_layer_units = hidden_layer_units
        self.nb_harmonics = nb_harmonics
        self.block_types= block_types
        self.stack = []
        self.thetas_dim = thetas_dim
        self.parameters = []
        self.device = device
        print('| N-Beats')

        for block_id in range(len(self.block_types)):
            block_init = NBeatsNet.select_block(self.block_types[block_id])
            block = block_init(ninp, self.hidden_layer_units, self.thetas_dim[block_id], self.device, self.backcast_length, self.forecast_length, block_type='fully_connected', nb_harmonics=self.nb_harmonics) #--> initialise your block in accordance with the type you've chosen just before
            self.parameters.extend(block.parameters())
            self.stack.append(block)
        print('| Initialization . . . .')    
        self.parameters = nn.ParameterList(self.parameters)
        
        self._opt = optim.Adam(self.parameters(),lr=1e-3,amsgrad=True)
        self._loss =F.l1_loss
        self.to(self.device)
        
    @staticmethod
    def select_block(block_type):
        if block_type == NBeatsNet.SEASONALITY_BLOCK:
            return SeasonalityBlock
        elif block_type == NBeatsNet.TREND_BLOCK:
            return TrendBlock
        else:
            return GenericBlock

    def fit(self, x_train, y_train, x_test, y_test, filename, epochs=10, batch_size=32):
        store_test_loss=np.zeros(epochs)
        store_loss=np.zeros(epochs)
        
       
        x_test_list=split(x_test, batch_size)
        y_test_list =split(y_test, batch_size)

        test_loss= self.evaluate(x_test, y_test, batch_size, filename, False)
        train_loss=self.evaluate(x_train, y_train, batch_size, filename, True)
        store_loss[0]=train_loss
        store_test_loss[0]=test_loss
        print('test loss--------- : {}'.format(test_loss))
        print('train loss-------- : {}'.format(train_loss))
        np.savetxt('./data/{}/train_loss.txt'.format(filename), store_loss)
        np.savetxt('./data/{}/val_loss.txt'.format(filename), store_test_loss)
           
        
        for epoch in range(1,epochs):
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
            

    def do_training(self, xtrain, ytrain) :
        self.train()
        total_loss=0.
        start_time=time.time()
            
        assert len(xtrain)==len(ytrain)


        for batch_id in range(len(xtrain)) :

            data, targets= xtrain[batch_id], ytrain[batch_id]
            self._opt.zero_grad()
            _, forecast = self(torch.tensor(data, dtype=torch.float).to(self.device))
           
            loss = self._loss(forecast,torch.tensor(targets, dtype=torch.float).to(self.device))
            total_loss+=loss.item()
            loss.backward()
            self._opt.step()
           
            log_interval = 50
            
            if batch_id % log_interval == 0 :
                if batch_id==0 :
                    cur_loss=total_loss
                else :
                    cur_loss = total_loss / log_interval
                print(' {:5d}/{:5d} batches | ''loss {:5.2f}'.format(batch_id, len(xtrain), cur_loss))

                total_loss=0

                      
    def evaluate(self, xtest, ytest, bsz, name, train, predict=False) :
        self.eval()
        xtest_list=split(xtest, bsz)
        ytest_list=split(ytest, bsz)
        assert len(xtest_list)==len(ytest_list)

        if predict :
            prediction=np.zeros_like(ytest)
        test_loss=[]
        for batch_id in range(len(xtest_list)) :
            data, targets = xtest_list[batch_id], ytest_list[batch_id]
            _,output=self(torch.tensor(data, dtype=torch.float).to(self.device))
            loss=self._loss(output, torch.tensor(targets, dtype=torch.float).to(self.device))
            test_loss.append(loss.item())
            if predict :
                print(output.shape)
                prediction[batch_id*output.shape[0]:(batch_id+1)*output.shape[0],:,:]=output.cpu().detach().numpy()
        if predict :
            if train :
                print('train')
                torch.save(prediction,'./data/{}/predictions_train.pt'.format(name))
            else :
                print('test')
                torch.save(prediction, './data/{}/predictions_test.pt'.format(name))
        return(np.mean(test_loss))  
        
                          
    def predict(self, x, return_backcast=False):
        self.eval()
        b, f = self(torch.tensor(x, dtype=torch.float).to(self.device))
        b, f = b.cpu().detach().numpy(), f.cpu().detach().numpy()
        if len(x.shape) == 3:
            b = np.expand_dims(b, axis=-1)
            f = np.expand_dims(f, axis=-1)
        if return_backcast:
            return b, f
        return f

    def forward(self, backcast):
        forecast = torch.zeros(size=(backcast.size()[0], self.forecast_length, backcast.size()[-1]))  # maybe batch size here.
        for block_id in range(len(self.stack)):
            b, f = self.stack[block_id](backcast)
            b=b.reshape((b.shape[0], self.backcast_length, self.ninp))
            f=f.reshape((f.shape[0], self.forecast_length, self.ninp))
            backcast = backcast.to(self.device) - b
            forecast = forecast.to(self.device) + f
        return backcast, forecast


def seasonality_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= thetas.shape[1], 'thetas_dim is too big.'
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    s1 = torch.tensor([np.cos(2 * np.pi * i * t) for i in range(p1)]).float()  # H/2-1
    s2 = torch.tensor([np.sin(2 * np.pi * i * t) for i in range(p2)]).float()
    S = torch.cat([s1, s2])
    return thetas.mm(S.to(device))


def trend_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= 4, 'thetas_dim is too big.'
    T = torch.tensor([t ** i for i in range(p)]).float()
    return thetas.mm(T.to(device))


def linear_space(backcast_length, forecast_length):
    ls = np.arange(-backcast_length, forecast_length, 1) / forecast_length
    b_ls = ls[:backcast_length]
    f_ls = ls[backcast_length:]
    return b_ls, f_ls


class Block(nn.Module):

    def __init__(self, ninp, units, thetas_dim, device, backcast_length=10, forecast_length=5, block_type='fully_connected', nb_harmonics=None):
        super(Block, self).__init__()
        self.units = units
        self.thetas_dim = thetas_dim
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.block_type=block_type
        if self.block_type=='fully_connected' :
            self.TFC=None
            self.fc1 = nn.Linear(backcast_length*ninp, units*ninp)
            self.fc2 = nn.Linear(units*ninp, units*ninp)
            self.fc3 = nn.Linear(units*ninp, units*ninp)
            self.fc4 = nn.Linear(units*ninp, units*ninp)
        else :
            self.TFC = TransformerModel(ninp,nhead=1, nhid=64, nlayers=1, backast_size=backcast_length, forecast_size=forecast_length, dropout=0.1, device=device)
            self.fc= nn.Linear(backcast_length*ninp, units*ninp)
        self.device = device
        self.backcast_linspace, self.forecast_linspace = linear_space(backcast_length, forecast_length)

        self.theta_b_fc = nn.Linear(units*ninp, thetas_dim*ninp, bias=False)
        self.theta_f_fc = nn.Linear(units*ninp, thetas_dim*ninp, bias=False)

    def forward(self, x):
        if self.block_type=='fully_connected' :
            x=x.reshape((x.shape[0], x.shape[1]*x.shape[2]))
            output = F.relu(self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x.to(self.device)))))))))
        else :
            x = x.transpose(0,1)
            y = self.TFC(x.to(self.device))
            y=y.transpose(0,1)
            y=y.reshape((y.shape[0], y.shape[1]*y.shape[2])) #remove last dim : block input=[bsz][length*ninp]
            output=F.relu(self.fc(y))
        return output

    def __str__(self):
        block_type = type(self).__name__
        return f'{block_type}(units={self.units}, thetas_dim={self.thetas_dim}, ' \
               f'backcast_length={self.backcast_length}, forecast_length={self.forecast_length}, ' \


class SeasonalityBlock(Block):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, block_type='fully_connected', nb_harmonics=None):
        if nb_harmonics:
            super(SeasonalityBlock, self).__init__(units, nb_harmonics, device, backcast_length,
                                                   forecast_length, block_type)
        else:
            super(SeasonalityBlock, self).__init__(units, forecast_length, device, backcast_length,
                                                   forecast_length, block_type)

    def forward(self, x):
        x = super(SeasonalityBlock, self).forward(x)
        # x= [128][80]
        backcast = seasonality_model(self.theta_b_fc(x), self.backcast_linspace, self.device)
        forecast = seasonality_model(self.theta_f_fc(x), self.forecast_linspace, self.device)
        
        return backcast, forecast


class TrendBlock(Block):

    def __init__(self, units, thetas_dim, device, backcast_length=10, forecast_length=5, nb_harmonics=None):
        super(TrendBlock, self).__init__(units, thetas_dim, device, backcast_length,
                                         forecast_length)

    def forward(self, x):
        x = super(TrendBlock, self).forward(x)
        backcast = trend_model(self.theta_b_fc(x), self.backcast_linspace, self.device)
        forecast = trend_model(self.theta_f_fc(x), self.forecast_linspace, self.device)
        return backcast, forecast


class GenericBlock(Block): # je pourrais faire theta_dims=theta_dims*ninp

    def __init__(self, ninp, units, thetas_dim, device, backcast_length=10, forecast_length=5, block_type='fully_connected', nb_harmonics=None):
        super(GenericBlock, self).__init__(ninp, units, thetas_dim, device, backcast_length, forecast_length, block_type)

        self.backcast_fc = nn.Linear(thetas_dim*ninp, backcast_length*ninp)
        self.forecast_fc = nn.Linear(thetas_dim*ninp, forecast_length*ninp)

    def forward(self, x):
        # no constraint for generic arch.
        x = super(GenericBlock, self).forward(x)

        theta_b = F.relu(self.theta_b_fc(x))
        theta_f = F.relu(self.theta_f_fc(x))
        backcast = self.backcast_fc(theta_b)  # generic. 3.3.
        forecast = self.forecast_fc(theta_f)  # generic. 3.3.

        return backcast, forecast
