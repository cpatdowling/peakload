import numpy as np
from scipy.stats import *
import copy

import cvxpy as cvx

#this configuration generates N samples and selects signal for reduction
#based on the maximum occurence

samples = []
num_samps = 20
budget_val = 1.0
means = np.array([2,1,1,1,2])
variances = np.array([1.0, 2.0, 1.0, 1.0, 1.0])

N = 2 #number of signals

for i in range(num_samps):
    samp = np.random.normal(means, variances, size=(5,))
    samples.append(samp)
    
samples = np.asarray(samples)

budget = np.expand_dims(np.array([budget_val for i in range(num_samps)]), 1)


print("Absolute maximums: ", np.max(samples, axis=0))
print("Expected Maximum: ", np.mean(np.max(samples, axis=1)))
print("Best reduced expected max: ", np.mean(np.max(samples - budget_val, axis=1)))
var_reduce = copy.copy(samples)
var_reduce[:,np.argmax(variances)] -= budget_val
print("Max Var reduced max: ", np.mean(np.max(var_reduce, axis=1)))
print("\n")

#selection = cvx.Variable((5,5),boolean=True)
selection = cvx.Variable((1,5),boolean=True)
max_val = cvx.Variable(num_samps)

constraint_1 = sum(sum(selection)) <= N #selection*np.array([budget for i in range(5)]) <= budget

constraints = []
for i in range(num_samps):
    for j in range(5):
        constraints.append(samples[i,j] <= max_val[i] - budget_val*selection[0,j])

utility = sum(max_val - budget*selection) #need to repeat selection so that all 2000
                                          #max val terms get every j'th item reduced by budget

problem = cvx.Problem(cvx.Minimize(utility), [constraint_1] + constraints)

problem.solve()

#print(dir(problem))
print("======Objective======")
print(problem.value)
print("\n")
print("======Solution=====")
print(selection.value)
print(sum(sum(selection.value)))
print("\n")
print("======Constraints======")
print("Sum", sum(samples - budget*selection).value)
print(problem.constraints[0].value())
print(problem.constraints[1].value())
print("\n")
