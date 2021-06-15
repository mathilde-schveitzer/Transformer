import os
import numpy as np
import math
import torch
import time
import random
import torch.nn as nn
import torch.nn.functional as F
from load_data import *

class MergedModel(nn.Module):

    def __init__(self, ninp,
                 nhead=2,
                 device=torch.device('cpu')):

        super(MergedModel, self).__init__()

        self.encoder = NbeatsNet(ninp, device=device)
        decoder_layers = TransformerDecoderLayers(ninp*nhead, nhead)
        self.decoder = TransformerDecoder(decoder_layers, num_layers=4)

        self.device=device

        
        self._opt = optim.Adam(self.parameters(), lr=1e-4, amsgrad=True)
        self._loss = F.l1_loss
        self.to(self.device)
        )

    def forward(self, input) :
        " premiere version : tgt = vide et on envoit dans memory la sortie de Nbeats
    TODO : essayer une version ou memory correspond a un assemblage des forecast predit au fur et a mesure et tgt serait la sortie finale reconstituee"
        _, memory = self.encoder(input) # [bsz, forecast_length, ninp]
        output=self.decoder(memory.transpose(0,1))
        
