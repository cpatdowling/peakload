import numpy as np
import datetime
import random
import json
import pickle

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
def expw_mae_loss(output, target):
    ret = ( (output - target).abs() * torch.exp(target) ).sum() / output.data.nelement()
    return(ret)
"""

def expw_mae_loss_(output, target, base=10.0):
    weights = Variable(torch.pow(base, target.data), requires_grad=False).cuda()
    ret = ( (output - target).abs() * weights ).sum() / output.data.nelement()
    return(ret)

def expw_mae_loss(output, target, base=10.0):
    base = Variable(torch.Tensor([base])).type_as(target)
    return(( (output - target).abs() * (torch.pow(base, target)) ).sum() / output.data.nelement() )

def batch_sample_list(datalist, batchsize):
    #list of training, target pair tuples
    remainder = len(datalist) % batchsize
    diff = batchsize - remainder
    tail = datalist[-diff:] + datalist[0:remainder]
    out = [ datalist[i*batchsize:(i+1)*batchsize] for i in range(int(float(len(datalist))/float(batchsize)))]
    out = out + [tail]
    return(out)

def torch_reshape_data(databatch):
    #flattens array inputs for a single list of training, target pairs
    inputs = []
    labels = []
    for sample in databatch:
        inputs.append(sample[0].flatten())
        labels.append(sample[1].flatten())
    return(torch.Tensor(np.asarray(inputs)), torch.Tensor(np.asarray(labels)))

def model_score(net_obj, datalist, threshes=[0.85, 0.95, 0.99, 0.999], batchsize=100):
    #model scoring is on L1 loss
    recs = {}
    precs = {}
    thresh_losses = {}
    for t in threshes:
        thresh_losses[t] = []
        recs[t] = []
        precs[t] = []
    set_losses = []
    
    #for also recording weighted loss function later
    criterion_l1 = nn.L1Loss()
    
    #do prediction on datalist
    inputs, labels = torch_reshape_data(datalist)
    labels = Variable(labels.cuda())
    inputs = Variable(inputs.cuda())
    outputs = net_obj(inputs)
    loss = criterion_l1(outputs, labels)
    set_losses.append(loss.data[0])
    
    labels = labels.data.cpu().numpy()
    outputs = outputs.data.cpu().numpy()
    
    for i in range(outputs.shape[0]):
        l = labels[i,:]
        o = outputs[i,:]
        
        for t in threshes:
            if any(l > t):
                rec = binary_rec(l, o, t)
                recs[t].append(rec)
            if any(o > t):
                prec = binary_prec(l, o, t)
                precs[t].append(prec)
  
            tl = thresh_loss(l, o, t)
            thresh_losses[t].append(np.nanmean(tl))
    
    for t in thresh_losses:
        thresh_losses[t] = np.nanmean(thresh_losses[t])
        if len(recs[t]) < 1:
            recs[t] = np.nan
        else:
            recs[t] = np.nanmean(recs[t])
        if len(precs[t]) < 1:
            precs[t] = np.nan
        else:
            precs[t] = np.nanmean(precs[t])

    mean_set_loss = np.nanmean(set_losses)
    
    ret = {"batch_recall": recs, "batch_precision": precs, "thresh_losses": thresh_losses, "mean_set_loss": mean_set_loss}          
    return(ret)

def batched_model_score(net_obj, datalist, threshes=[0.85, 0.95, 0.99, 0.999], batchsize=100):
    recs = {}
    precs = {}
    thresh_losses = {}
    for t in threshes:
        thresh_losses[t] = []
        recs[t] = []
        precs[t] = []
    set_losses = []
    
    #batchdata
    batches = batch_sample_list(datalist, batchsize)
    
    #for also recording weighted loss function later
    criterion_l1 = nn.L1Loss()
    
    #do prediction on datalist
    for i, data, in enumerate(batches):
        inputs, labels = torch_reshape_data(data)
        labels = Variable(labels.cuda())
        inputs = Variable(inputs.cuda())
        outputs = net_obj(inputs)
        loss = criterion_l1(outputs, labels)
        set_losses.append(loss.data[0])
        
        labels = labels.data.cpu().numpy()
        outputs = outputs.data.cpu().numpy()
        
        for i in range(outputs.shape[0]):
            l = labels[i,:]
            o = outputs[i,:]
            
            for t in threshes:
                if any(l > t):
                    rec = binary_rec(l, o, t)
                    recs[t].append(rec)
                if any(o > t):
                    prec = binary_prec(l, o, t)
                    precs[t].append(prec)    
                tl = thresh_loss(l, o, t)
                thresh_losses[t].append(np.nanmean(tl))
    
    for t in thresh_losses:
        thresh_losses[t] = np.nanmean(thresh_losses[t])
        if len(recs[t]) < 1:
            recs[t] = np.nan
        else:
            recs[t] = np.nanmean(recs[t])
        if len(precs[t]) < 1:
            precs[t] = np.nan
        else:
            precs[t] = np.nanmean(precs[t])

    mean_set_loss = np.nanmean(set_losses)
    
    ret = {"batch_recall": recs, "batch_precision": precs, "thresh_losses": thresh_losses, "mean_set_loss": mean_set_loss}          
    return(ret)

def print_model_scores(scores, name):
    print("====" + name + "====")
    for k in scores:
        print(k + ": " + str(scores[k]))
    print("=================\n")
    
    
def binary_rec(labels, pred, thresh=0.85):
    #returns nan if there are no positives above the threshold
    pred_pos = 0 #values I've predicted above a threshold (numerator)
    all_pos = 0 #all true positives in test data (denominator)
    for i in range(len(labels)):
        if labels[i] >= thresh:
            all_pos += 1
            if pred[i] >= thresh:
                pred_pos += 1
    if all_pos == 0:
        return(np.nan)
    else:
        return(float(pred_pos)/float(all_pos))
    
def binary_prec(labels, pred, thresh=0.85):
    #returns nan if no predictions above the threshold were made
    pred_pos = 0 #values I've predicted above a threshold (denominator)
    true_pos = 0 #positives I've correctly predicted (numerator)
    for i in range(len(pred)):
        if pred[i] >= thresh:
            pred_pos += 1
            if labels[i] >= thresh:
                true_pos += 1
    if pred_pos == 0:
        return(np.nan)
    else:
        return(float(true_pos)/float(pred_pos))

def thresh_loss(labels, pred, thresh=0.85):
    #returns nan if neither predictions or true values are above
    #threshold
    out = []
    for i in range(len(labels)):
        if pred[i] >= thresh or labels[i] >= thresh:
            loss = np.abs(labels[i] - pred[i])
            out.append(loss)
    if len(out) == 0:
        out = np.nan
    else:
        pass
    return(np.nanmean(out))
    
def eval_perf_dropoff(labels, pred):
    thresh_losses = np.arange(0.85,1,0.01)
    
    out_prec = []
    out_rec = []
    out_loss = []
    
    for t in thresh_losses:
        out_prec.append(binary_prec(labels, pred, t))
        out_rec.append(binary_rec(labels, pred, t))
        out_loss.append(thresh_loss(labels, pred, t))
        
    return(out_prec, out_rec, out_loss)

def f1_score(rec, prec):
    return(2.0/((1/rec) + (1/prec)))

def set_loss(net_obj, data):
    inputs, labels = torch_reshape_data(data)
    labels = Variable(labels.cuda())
    outputs = net_obj(Variable(inputs.cuda()))
    criterion_l1 = nn.L1Loss()
    loss = criterion_l1(outputs, labels)
    return(loss.data[0])
