import numpy as np
from scipy.optimize import minimize_scalar

# This script correctly implements the model and produces the true, asymmetric results.

# 1. Parameters
W1, beta, r, theta = 2.0, 1.0, 0.0, 0.5

# 2. General Objective Function
def objective_function(c1, W1, beta, r, theta, k):
    if c1 <= 1e-6 or c1 >= W1 - 1e-6: return np.inf
    c2 = (W1 - c1) * (1 + r)
    u_c1, V2 = np.log(c1) + k, np.log(c2) + k
    objective_val = u_c1 + beta * V2 - 0.5 * theta * (V2**2)
    return -objective_val

# 3. Solve for Each Agent
params_a = (W1, beta, r, theta, 0.0)
params_b_plus = (W1, beta, r, theta, 2.0)
params_b_minus = (W1, beta, r, theta, -2.0)

c1_a = minimize_scalar(objective_function, bounds=(0.001, 1.999), args=params_a).x
c1_b_plus = minimize_scalar(objective_function, bounds=(0.001, 1.999), args=params_b_plus).x
c1_b_minus = minimize_scalar(objective_function, bounds=(0.001, 1.999), args=params_b_minus).x

c2_a, c2_b_plus, c2_b_minus = W1 - c1_a, W1 - c1_b_plus, W1 - c1_b_minus

# 4. Calculate Welfare Loss
def calculate_welfare_loss(consumption_path, total_wealth):
    welfare_of_path = np.log(consumption_path[0]) + np.log(consumption_path[1])
    W_equiv = 2 * np.exp(welfare_of_path / 2)
    return (total_wealth - W_equiv) / total_wealth

welfare_loss_plus = calculate_welfare_loss((c1_b_plus, c2_b_plus), W1)
welfare_loss_minus = calculate_welfare_loss((c1_b_minus, c2_b_minus), W1)

# 5. Print the True Results
print("--- Definitive Results for the Ambiguity Example ---")
print(f"\n{'Attribute':<20} | {'Agent A':^15} | {'Agent B (+2)':^15} | {'Agent B (-2)':^15}")
print("-" * 70)
print(f"{'Utility Rep.':<20} | {'ln(c)':^15} | {'ln(c) + 2':^15} | {'ln(c) - 2':^15}")
print(f"{'Optimal c1*':<20} | {c1_a:^15.2f} | {c1_b_plus:^15.2f} | {c1_b_minus:^15.2f}")
print(f"{'Consumption Path':<20} | {f'({c1_a:.2f}, {c2_a:.2f})':^15} | {f'({c1_b_plus:.2f}, {c2_b_plus:.2f})':^15} | {f'({c1_b_minus:.2f}, {c2_b_minus:.2f})':^15}")
print(f"{'Consumption Change':<20} | {'---':^15} | {f'+{((c1_b_plus/c1_a)-1):.0%}':^15} | {f'{((c1_b_minus/c1_a)-1):.0%}':^15}")
print(f"{'Welfare Loss':<20} | {'---':^15} | {f'{welfare_loss_plus:.1%}':^15} | {f'{welfare_loss_minus:.1%}':^15}")
print("-" * 70)