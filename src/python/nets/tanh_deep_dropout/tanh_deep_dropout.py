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
        self.H4 = params['HIDDEN_4']
        self.H5 = params['HIDDEN_5']
        self.H6 = params['HIDDEN_6']
        self.D_out = params['OUTPUT_DIM']
        self.drop = nn.Dropout(p=0.2)
        self.l1 = nn.Linear(self.D_in, self.H1)
        self.l2 = nn.Linear(self.H1, self.H2)
        self.l3 = nn.Linear(self.H2, self.H3)
        self.l4 = nn.Linear(self.H3, self.H4)
        self.l5 = nn.Linear(self.H4, self.H5)
        self.l6 = nn.Linear(self.H5, self.H6)
        self.l7 = nn.Linear(self.H6, self.D_out)
    
    def forward(self, x):
        x = F.sigmoid(self.l1(x))
        x = self.drop(x)
        x = F.tanh(self.l2(x))
        x = F.sigmoid(self.l3(x))
        x = F.logsigmoid(self.l4(x))
        x = F.sigmoid(self.l5(x))
        x = F.tanh(self.l6(x))
        x = self.l7(x)
        return(x)
