import numpy as np
import random
import warnings
import json

# This is the definitive script. It uses the exact logic from the successful
# two-region plotting script, but constrains its search *specifically* to the
# most interesting region: the High-EIS region (alpha > 14, rho < 0.4).
# It will find and log the first verifiable example from this target zone.

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Using the simple, stable iterative solver that we know works ---
def real_power(base, exp):
    if np.iscomplex(base) or np.iscomplex(exp): return np.nan
    if base < 0: return -(np.abs(base) ** exp)
    return base ** exp

def u_A(c, alpha):
    return (c**(1 - alpha)) / (1 - alpha)

def u_B(c, alpha):
    return ((c**(1 - alpha) - 1)) / (1 - alpha)

def solve_V_fight_iterative(utility_func, params):
    beta, alpha, rho, c_war, c_mono, p = params['beta'], params['alpha'], params['rho'], params['c_war'], params['c_mono'], params['p']
    u_war, V_mono = utility_func(c_war, alpha), utility_func(c_mono, alpha)
    if np.isinf(u_war) or np.isinf(V_mono): return np.nan
    
    v_guess = u_war
    for _ in range(250):
        try:
            ce_term = p * real_power(V_mono, 1 - alpha) + (1 - p) * real_power(v_guess, 1 - alpha)
            if np.isnan(ce_term): return np.nan
            certainty_equivalent = real_power(ce_term, 1 / (1 - alpha))
            value_term = (1 - beta) * real_power(u_war, 1 - rho) + beta * real_power(certainty_equivalent, 1 - rho)
            v_next = real_power(value_term, 1 / (1 - rho))
            if np.isnan(v_next) or np.isinf(v_next): return np.nan
            if abs(v_next - v_guess) < 1e-12: return v_next
            v_guess = v_next
        except (ValueError, TypeError, ZeroDivisionError):
            return np.nan
    return v_guess # Return the converged value

def find_and_log_high_eis_success(grid_size=50, randomization_trials=30):
    print("Searching specifically in the High-EIS region (alpha > 14, rho < 0.4)...")
    # --- TARGETED SEARCH RANGES ---
    alphas = np.linspace(14.0, 20.0, grid_size)
    rhos = np.linspace(0.1, 0.4, grid_size)
    
    total_trials = 0
    for i in range(grid_size):
        alpha = alphas[i]
        for j in range(grid_size):
            rho = rhos[j]
            # No need for a progress bar here, as it should be fast
            for _ in range(randomization_trials):
                total_trials += 1
                params = {
                    'beta': 0.95, 'alpha': alpha, 'rho': rho,
                    'c_exit': random.uniform(0.8, 0.98),
                    'c_war': random.uniform(0.7, 0.98),
                    'c_mono': random.uniform(1.5, 7.0),
                    'p': random.uniform(0.05, 0.30)
                }
                if params['c_war'] >= params['c_exit']: continue

                V_A_Exit = u_A(params['c_exit'], alpha)
                V_A_Fight = solve_V_fight_iterative(u_A, params)
                if np.isnan(V_A_Fight) or V_A_Fight >= V_A_Exit:
                    continue

                V_B_Exit = u_B(params['c_exit'], alpha)
                V_B_Fight = solve_V_fight_iterative(u_B, params)
                if np.isnan(V_B_Fight) or V_B_Fight <= V_B_Exit:
                    continue
                
                # --- SUCCESS ---
                print(f"\nSUCCESS! Found a verifiable set after {total_trials} total trials.\n")
                
                results = {
                    'params': params,
                    'V_A_Fight': V_A_Fight, 'V_A_Exit': V_A_Exit,
                    'V_B_Fight': V_B_Fight, 'V_B_Exit': V_B_Exit
                }
                with open('successful_high_eis_parameters.json', 'w') as f:
                    json.dump(results, f, indent=4)
                print("Parameters and results saved to 'successful_high_eis_parameters.json'")
                
                return results

    print("\nSearch complete. No flip was found in this run.")
    return None

def print_report(results):
    if not results: return
    params = results['params']
    V_A_F, V_A_E = results['V_A_Fight'], results['V_A_Exit']
    V_B_F, V_B_E = results['V_B_Fight'], results['V_B_Exit']
    decision_A = "Exit" if V_A_F < V_A_E else "Fight"
    decision_B = "Exit" if V_B_F < V_B_E else "Fight"
    
    print("\n--- Verification Report ---")
    print("\n--- Parameters ---")
    for key, val in params.items():
        print(f"  {key}: {val:.4f}")
        
    print("\n" + "="*50)
    print(f"{'Valuation':<20} {'Agent A (Admissible)':<25} {'Agent B (Inadmissible)':<25}")
    print("-"*70)
    print(f"{'V(Exit)':<20} {V_A_E:<25.4f} {V_B_E:<25.4f}")
    print(f"{'V(Fight)':<20} {V_A_F:<25.4f} {V_B_F:<25.4f}")
    print("-"*70)
    print(f"{'Optimal Strategy':<20} {decision_A:<25} {decision_B:<25}")
    print("="*50 + "\n")

if __name__ == '__main__':
    results = find_and_log_high_eis_success()
    print_report(results)

