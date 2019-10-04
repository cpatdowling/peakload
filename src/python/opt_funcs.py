import numpy as np
from scipy.stats import *
import copy
import sys

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

if __name__ == "__main__":
    means = np.loadtxt(sys.argv[1], delimiter=",")
    variances = np.loadtxt(sys.argv[2], delimiter=",")
    T = len(means)
    results = np.zeros((3,T))

    budget_val = 1.0
    num_samps = 100
    num_signals = 2
    budget_max = num_signals*budget_val

    samples = []
    for i in range(num_samps):
        samp = np.random.normal(means, variances, size=(T,))
        samples.append(samp)
        
    samples = np.asarray(samples)
    
    print("======Sample Data======")
    print("Number of time periods: ", T)
    print("Absolute maximums: ", np.max(samples, axis=0))
    print("Absolute minimums: ", np.min(samples, axis=0))
    print("Expected maximum: ", np.mean(np.max(samples, axis=1)))
    print("Best reduced expected max: ", np.mean(np.max(samples - budget_val, axis=1)))
    print("\n")
    
    #integer signals
    print("Minimizing expected max w/ integer signals...\n")
    selection = cvx.Variable(T,boolean=True)
    max_val = cvx.Variable(num_samps)
    
    constraints = [sum(selection) <= num_signals]
    for i in range(num_samps):
        for j in range(T):
            constraints.append(max_val[i] + budget_val*selection[j] >= samples[i,j])
            
    utility = (1.0/float(num_samps))*sum(max_val)
    
    problem_int = cvx.Problem(cvx.Minimize(utility), constraints)
    
    problem_int.solve(solver=cvx.ECOS_BB)
    
    print("======Objective======")
    print(problem_int.value)
    print("\n")
    print("======Solution======")
    print(selection.value)
    print(sum(selection.value))
    print("\n")
    
    results[0,:] = selection.value
    
    #choosing by modes of argmax
    print("Minimizing expected max by allocating integer signals to modes of argmax...\n")
    samples_copy = copy.copy(samples)
    
    modes = all_modes(np.argmax(samples, axis=1)) #sorted list of most common max round
    solution_mode = np.zeros((T,))
    for m in range(num_signals):
        samples_copy[:,modes[m]] -= budget_val
        solution_mode[modes[m]] = 1
        
    objective = np.mean(np.max(samples_copy, axis=1))
    
    print("======Objective======")
    print(objective)
    print("\n")
    print("======Solution======")
    print(solution_mode)
    print("\n")
    
    results[1,:] = selection.value
    
    
    #continuous signals
    print("Minimizing expected max w/ continuous *curtailment* signals only...\n")
    selection = cvx.Variable(T)
    max_val = cvx.Variable(num_samps)
    
    constraints = [selection >= 0, sum(selection) <= budget_max]
    for i in range(num_samps):
        for j in range(T):
            constraints.append(max_val[i] + budget_val*selection[j] >= samples[i,j])
            
    utility = (1.0/float(num_samps))*sum(max_val)
    
    problem_lin = cvx.Problem(cvx.Minimize(utility), constraints)
    
    problem_lin.solve(solver=cvx.ECOS_BB)
    
    print("======Objective======")
    print(problem_lin.value)
    print("\n")
    print("======Solution======")
    print(selection.value)
    print(sum(selection.value))
    print("\n")
    
    results[2,:] = selection.value
    
    with open(sys.argv[3], 'w') as d:
        np.savetxt(d, np.around(results, 4), delimiter=",")
    
    
