import numpy as np
import matplotlib.pyplot as plt
import random
import warnings
from tqdm import tqdm

# This is the definitive plotting script. It focuses the search on the
# High-EIS (rho < 1) region to create the final, correct plot.

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Using the simple, stable iterative solver ---
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
    return np.nan # Did not converge

def generate_plot_data(grid_size=50, randomization_trials=30):
    print("Starting focused grid search in the High-EIS (rho < 1) region...")
    alphas = np.linspace(3.0, 20, grid_size)
    rhos = np.linspace(0.1, 1.0, grid_size) # FOCUS THE GRID
    results_grid = np.zeros((grid_size, grid_size))

    for i, alpha in enumerate(tqdm(alphas, desc="Alpha Progress")):
        for j, rho in enumerate(rhos):
            flip_found = False
            for _ in range(randomization_trials):
                params = {
                    'beta': 0.95, 'alpha': alpha, 'rho': rho,
                    'c_exit': random.uniform(0.8, 0.98),
                    'c_war': random.uniform(0.7, 0.98),
                    'c_mono': random.uniform(1.5, 7.0),
                    'p': random.uniform(0.05, 0.30)
                }
                if params['c_war'] >= params['c_exit']: continue

                V_A_Fight = solve_V_fight_iterative(u_A, params)
                if np.isnan(V_A_Fight): continue
                V_A_Exit = u_A(params['c_exit'], alpha)
                if V_A_Fight >= V_A_Exit: continue

                V_B_Fight = solve_V_fight_iterative(u_B, params)
                if np.isnan(V_B_Fight): continue
                V_B_Exit = u_B(params['c_exit'], alpha)
                if V_B_Fight > V_B_Exit:
                    flip_found = True
                    break
            
            if flip_found:
                results_grid[j, i] = 1

    return alphas, rhos, results_grid

def create_plot(alphas, rhos, grid):
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 6.5))
    c = ax.pcolormesh(alphas, rhos, grid, cmap='viridis', shading='auto')
    
    ax.set_xlabel(r'Coefficient of Relative Risk Aversion ($\alpha$)', fontsize=12)
    ax.set_ylabel(r'Inverse EIS ($\rho$)', fontsize=12)
    ax.set_title('Region of Equilibrium Flip for High-EIS Agents', fontsize=14, pad=15)
    ax.axis([alphas.min(), alphas.max(), rhos.min(), rhos.max()])
    
    cbar = fig.colorbar(c, ticks=[0, 1], orientation='vertical')
    cbar.ax.set_yticklabels(['No Flip Occurs', 'Equilibrium Flip Occurs'], fontsize=10)
    
    plt.tight_layout()
    plt.savefig('robustness_plot_high_eis.png', dpi=300)
    print("Plot saved as 'robustness_plot_high_eis.png'")

if __name__ == '__main__':
    alphas, rhos, grid_data = generate_plot_data()
    create_plot(alphas, rhos, grid_data)
