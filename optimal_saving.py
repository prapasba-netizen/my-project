import numpy as np
from scipy.optimize import brentq

# 1. Parameters
beta = 0.96
alpha = 5.0
eis = 1.5
rho = 1 / eis
r = 0.04

# 2. Utility Functions
def u_admissible(c, alpha):
    return (c**(1 - alpha)) / (1 - alpha)
def u_prime_admissible(c, alpha):
    return c**(-alpha)
def u_inadmissible(c, alpha):
    return (c**(1 - alpha) - 1) / (1 - alpha)
def u_prime_inadmissible(c, alpha):
    return c**(-alpha)

# 3. Correct Epstein-Zin First-Order Condition Solver
def ez_foc_solver(s, u_func, u_prime_func, params):
    beta, r, rho, alpha = params
    if s <= 0.001 or s >= 0.999: return np.inf
    c1, c2 = 1 - s, s * (1 + r)
    
    u_val1, u_val2 = u_func(c1, alpha), u_func(c2, alpha)
    u_prime1, u_prime2 = u_prime_func(c1, alpha), u_prime_func(c2, alpha)
    
    # Correct FOC: (1-beta)*u(c1)**(-rho)*u'(c1) = beta*(1+r)*u(c2)**(-rho)*u'(c2)
    lhs = (1 - beta) * (abs(u_val1))**(-rho) * u_prime1
    rhs = beta * (1 + r) * (abs(u_val2))**(-rho) * u_prime2
    
    return lhs - rhs

# 4. Solve and Report Correct Numbers
params = (beta, r, rho, alpha)
s_admissible_correct = brentq(ez_foc_solver, 0.01, 0.99, args=(u_admissible, u_prime_admissible, params))
s_inadmissible_correct = brentq(ez_foc_solver, 0.01, 0.99, args=(u_inadmissible, u_prime_inadmissible, params))

# 5. Print the theoretically consistent results
print("--- Theoretically Consistent Results for Section 3.3 ---")
print("\nThese are the numbers that should be reported in the paper to ensure consistency")
print("between the model description and the quantitative example.\n")

print(f"Scenario A (Admissible): Correct Optimal Savings Rate = {s_admissible_correct:.1%}")
print(f"Scenario B (Inadmissible): Correct Optimal Savings Rate = {s_inadmissible_correct:.1%}")
print(f"\nThe difference is {(s_admissible_correct - s_inadmissible_correct)*100:.1f} percentage points.")