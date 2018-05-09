import torch
import torch.nn as nn
import torch.nn.functional as F

class nnet(nn.Module):
    def __init__(self, params):
        super(nnet, self).__init__()
        self.D_in = params['LOOKBACK'] * params['FEATURE_DIM']
        self.H1 = params['HIDDEN_1']
        self.D_out = params['OUTPUT_DIM']
        self.l1 = nn.Linear(self.D_in, self.H1)
        self.l2 = nn.Linear(self.H1, self.D_out)
    
    def forward(self, x):
        x = F.sigmoid(self.l1(x))
        x = self.l2(x)
        return(x)
