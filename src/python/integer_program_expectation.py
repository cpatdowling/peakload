import numpy as np
import scipy.stats
import copy

import cvxpy as cvx

def all_modes(arr):
    out = {}
    for i in range(len(arr)):
        if arr[i] not in out:
            out[arr[i]] = 1
        else:
            out[arr[i]] += 1
    sorted_modes = [tup[0] for tup in sorted(out.items(), key = lambda x : x[1], reverse=True)]
    return(sorted_modes)

#this configuration generates N samples and selects signal for reduction
#based on the expected maximum

samples = []
num_samps = 2000
budget_val = 1.0
means = np.array([1,1,1,1,2])
variances = np.array([1.0, 100.0, 1.0, 1.0, 1.0])

for i in range(num_samps):
    samp = np.random.normal(means, variances, size=(5,))
    samples.append(samp)
    
samples = np.asarray(samples)

budget = np.expand_dims(np.array([budget_val for i in range(num_samps)]), 1)


print("Absolute maximums: ", np.max(samples, axis=0))
print("Expected max over t: ", np.mean(np.max(samples,axis=1)))
print("Mode Maximum t: ", scipy.stats.mode(np.argmax(samples, axis=1)))
print("Best possible reduced expected max: ", np.mean(np.max(samples - budget_val, axis=1)))
var_reduce = copy.copy(samples)
var_reduce[:,np.argmax(variances)] -= budget_val
print("Max Var reduced max: ", np.mean(np.max(var_reduce, axis=1)))
print("\n")


N = 2

#go through mode in descending order according to N
modes = all_modes(np.argmax(samples, axis=1))

i = 0
while i < N:
    m = modes[i]
    samples[:,i] -= budget_val
    i += 1

print("======Objective======")
print(np.mean(np.max(samples,axis=1)))
print("\n")
print("======Solution=====")
i = 0
selection = np.zeros((5,))
while i < N:
    m = modes[i]
    selection[m] = 1
    i += 1
print(selection)
print("\n")
print("======Constraints======")
print("Budget signal limit, ", N)
print(np.sum(selection))
