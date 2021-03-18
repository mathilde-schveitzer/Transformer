import math
import torch
import time
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, bptt, dropout=0.5, device=torch.device('cpu')):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder=nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.device = device
        self.criterion=None
        self.optimizer=torch.optim.SGD(self.parameters(),lr=5)
        self.scheduler=None
        self.init_weights()
        print('|T R A N S F O R M E R : Optimus Prime is ready |')
        
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


#TODO : bouger ce truc #
    def train(self, train_data):
        self.train() # Turn on the train mode
        total_loss = 0.
        start_time = time.time()
        src_mask = TransformerModel.generate_square_subsequent_mask(self.bptt).to(self.device)
        for batch, i in enumerate(range(0, train_data.size(0) - 1, self.bptt)):
            data, targets = get_batch(train_data, i)
            self.optimizer.zero_grad()
            if data.size(0) != bptt :
                src_mask = TransformerModel.generate_square_subsequent_mask(data.size(0)).to(self.device)
                output = self(data, src_mask)
                loss = self.criterion(output.view(-1, self.ntokens), targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                self.optimizer.step()

            total_loss += loss.item()
            log_interval = 200
            if batch % log_interval == 0 and batch > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | '
                      'lr {:02.2f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f}'.format(
                          epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                          elapsed * 1000 / log_interval,
                          cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()

    def evaluate(self, data_source):
        self.eval(data_source) # Turn on the evaluation mode
        total_loss = 0.
        src_mask = TransformerModel.generate_square_subsequent_mask(self.bptt).to(self.device)
        with torch.no_grad():
            for i in range(0, data_source.size(0) - 1, self.bptt):
                data, targets = get_batch(data_source, i)
                if data.size(0) != bptt:
                    src_mask = TransformerModel.generate_square_subsequent_mask(data.size(0)).to(self.device)
                    output = self(data, src_mask)
                    output_flat = output.view(-1, self.ntokens)
                    total_loss += len(data) * self.criterion(output_flat, targets).item()
        return total_loss / (len(data_source) - 1)

    def fit(self, train_data, val_data, epochs):
        best_val_loss=float("inf")
        best_model=None
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            self.train(train_data)
            val_loss = self.evaluate(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(val_loss)))
            print('-' * 89)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
        
            self.scheduler.step()
        return(best_model, best_val_loss)

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

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
