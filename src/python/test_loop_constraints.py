import cvxpy as cvx

samples = 10
x = cvx.Variable(samples)
y = range(1, samples+1)
constraints = []

for i in range(samples):
    constraints += [
        y[i] * x[i] <= 1000,
        x[i] >= i
    ]

objective = cvx.Maximize(cvx.sum(x))

print(constraints)

prob = cvx.Problem(objective, constraints)
prob.solve()
print(x.value)
