import numpy as np
import datetime
import random
import json
import pickle
import sys
import os

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import importlib

sys.path.append("./")
from utils import *

def train(net_obj, loss_fnc, opt_fnc, traindata, valdata, batchsize=100, epochs=50):  
    train_batches = batch_sample_list(traindata, batchsize)
    num_batches = len(train_batches)
    
    train_epoch_set_loss = []
    val_epoch_set_loss = []
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_batches):
            inputs, labels = torch_reshape_data(data)

            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda()) #they need to be .cuda() with each training epoch

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net_obj(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]

            #training loss
            #print("epoch " + str(epoch) + " ,iter " + str(i) + ": " + str(running_loss))
            #training_epoch_loss.append(running_loss)
        
        print("==epoch " + str(epoch) + "==")
        train_samp = random.randint(0,num_batches)
        train_set_loss = set_loss(net_obj, train_batches[train_samp-1])
        train_epoch_set_loss.append(train_set_loss)
        print("training set sampled loss: " + str(train_set_loss))
        val_set_loss = set_loss(net_obj, valdata)
        val_epoch_set_loss.append(val_set_loss)
        print("validation set loss: " + str(val_set_loss))
       
    #sampled epoch set loss
    print("\n")
    print("Model results")
    
    train_model_scores = batched_model_score(net_obj, traindata) #BATCHED/AVERAGED SCORES
    print_model_scores(train_model_scores, "training data scores")
    
    val_model_scores = model_score(net_obj, valdata)
    print_model_scores(val_model_scores, "validation data scores")
                
    return({'train_epoch_set_loss':train_epoch_set_loss, 'val_epoch_set_loss':val_epoch_set_loss, 'train_model_scores':train_model_scores, 'val_model_scores':val_model_scores})
    

if __name__ == "__main__":
    random.seed(45) #fixed for all experiments, my high school football jersey number
    torch.manual_seed(45)
    
    #command line args
    net_class_path = sys.argv[1]
    params_path = sys.argv[2]
    train_data_path = sys.argv[3]
    test_data_path = sys.argv[4]
    model_ident = sys.argv[5]
    
    #get parameters for network
    with open(params_path, 'rb') as p:
        params = pickle.load(p)
  
    #load formatted data (data must be formatted to output/lookback parameters
    if params['DATA'] == 'directory':
        trainfiles = os.listdir(train_data_path)
        normed_training_data_pairs = []
        for f in trainfiles:
            with open(train_data_path + "/" + f, 'rb') as d:
                data = pickle.load(d)
                normed_training_data_pairs.append( (np.nan_to_num(data[0]), np.nan_to_num(data[1])) )
        testfiles = os.listdir(test_data_path)
        normed_test_data_pairs = []
        for f in testfiles:
            with open(test_data_path + "/" + f, 'rb') as d:
                data = pickle.load(d)
                normed_test_data_pairs.append( (np.nan_to_num(data[0]), np.nan_to_num(data[1])) )
    elif params['DATA'] == 'file':
        with open(train_data_path, 'rb') as d:
            normed_training_data_pairs = pickle.load(d)
        with open(test_data_path, 'rb') as d:
            normed_test_data_pairs = pickle.load(d)
    print("Data loaded")
    
    #apply data and parameter values to network architecture
    arch = []
    for dim in params:
        arch.append(dim + "_" + str(params[dim]))
    params['FEATURE_DIM'] = len(normed_test_data_pairs[0][0]) #columns to be flattened
    print("Feature dimension: " + str(params['FEATURE_DIM']))
    params['OUTPUT_DIM'] = len(normed_test_data_pairs[1][1]) #vector length
    print("Output dimension: " + str(params['OUTPUT_DIM']))
    
    batch_size = params['BATCH_SIZE'] #batch size

    #shuffle training/test data pairs, split out validation
    random.shuffle(normed_training_data_pairs)
    training_pairs = normed_training_data_pairs[0:int(0.9*float(len(normed_training_data_pairs)))]
    val_pairs = normed_training_data_pairs[int(0.9*float(len(normed_training_data_pairs))):]
    random.shuffle(normed_test_data_pairs)

    #accept filename as command line argument, instantiate class here
    #class requires parameter inputs, hardcode for now
    sys.path.append(net_class_path)
    print("Importing net class from: " + net_class_path)
    mod = net_class_path.split("/")[-1][:-3] #get filename, strip .py extension
    netclass = importlib.import_module(mod)
    
    #instantiate network
    net = netclass.nnet(params).cuda()
    print("Instantiating network " + mod + ", with architecture " + " ".join(arch))

    if params['LOSS'] == "L1":
        criterion = nn.L1Loss()
    elif params['LOSS'] == "EW":
        criterion = expw_mae_loss #import from utils
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #0.001 before switch to 1.0 - alpha^{F(x)}
    
    scores = train(net, criterion, optimizer, training_pairs, val_pairs, epochs=params['EPOCHS'])
    
    test_model_scores = model_score(net, normed_test_data_pairs)
    scores['test_model_scores'] = test_model_scores
    print_model_scores(test_model_scores, "test data scores")
    
    
    #use model_ident for save file name
    torch.save(net, "/home/chase/projects/peakload/data/nets/models/" + model_ident +  ".pt")
        
    with open("/home/chase/projects/peakload/data/nets/reports/model_registry.txt", 'a') as f:
        f.write(model_ident + "," + train_data_path.split("/")[0] + "," + ",".join(arch) + "\n")
        
    with open("/home/chase/projects/peakload/data/nets/reports/" + model_ident + "_train_test_scores_.pck", 'wb') as p:
        pickle.dump(scores, p)


    
