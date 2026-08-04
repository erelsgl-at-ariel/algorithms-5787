#!python3

"""
Demonstration of the fractional-budgeting algorithm maximizing the product of utilities.

AUTHOR: Erel Segal-Halevi
SINCE:  2019-11
"""


import cvxpy

def allocate(preferences_matrix:list[list], budget:int):
    num_of_voters = len(preferences_matrix)
    num_of_issues = len(preferences_matrix[0])
    allocations = cvxpy.Variable(num_of_issues)
    utilities = [allocations @ preferences_matrix[i] for i in range(num_of_voters)]
    donations = [budget/num_of_voters for _ in range(num_of_voters)]

    sum_of_logs = cvxpy.sum([cvxpy.log(u) for u in utilities])
    positivity_constraints = [v >= 0 for v in allocations]
    sum_constraint = [cvxpy.sum(allocations)==sum(donations)]

    problem = cvxpy.Problem(
        cvxpy.Maximize(sum_of_logs),
        constraints = positivity_constraints+sum_constraint)
    problem.solve()


# There are 4 projects, denoted by a, b, c, d.
# The same letters denote the budget allocated to them.
allocations = cvxpy.Variable(3)
a, b, c = allocations

# There are 5 citizens. Their preferences are: ab, ac, ad, bc, a. The total budget is 500 - 100 for each citizen
donations = [3, 3, 3, 3]
utilities = [a, a+b, a+c, b]

sum_of_logs = cvxpy.sum([cvxpy.log(u) for u in utilities])
positivity_constraints = [v >= 0 for v in allocations]
sum_constraint = [cvxpy.sum(allocations)==sum(donations)]

problem = cvxpy.Problem(
    cvxpy.Maximize(sum_of_logs),
    constraints = positivity_constraints+sum_constraint)
problem.solve()

utility_values = [u.value for u in utilities]
print("BUDGET: a={}, b={}, c={}".format(a.value, b.value, c.value))
print("UTILS : {} + {} + {} + {} = {}".format(*utility_values, sum(utility_values)))
