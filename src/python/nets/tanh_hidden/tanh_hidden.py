import torch
import torch.nn as nn
import torch.nn.functional as F

class nnet(nn.Module):
    def __init__(self, params):
        super(nnet, self).__init__()
        self.D_in = params['FEATURE_DIM'] #params['LOOKBACK'] * params['FEATURE_DIM'], previously flattening on the fly
        self.H1 = params['HIDDEN_1']
        self.H2 = params['HIDDEN_2']
        self.H3 = params['HIDDEN_3']
        self.D_out = params['OUTPUT_DIM']
        self.l1 = nn.Linear(self.D_in, self.H1)
        self.l2 = nn.Linear(self.H1, self.H2)
        self.l3 = nn.Linear(self.H2, self.H3)
        self.l4 = nn.Linear(self.H3, self.D_out)
    
    def forward(self, x):
        x = F.sigmoid(self.l1(x))
        x = F.tanh(self.l2(x))
        x = F.sigmoid(self.l3(x))
        x = self.l4(x)
        return(x)
