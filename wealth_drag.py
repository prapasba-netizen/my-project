import numpy as np

# --------------------------------------------------------------------------
# 1. DEFINE MODEL PARAMETERS (from Appendix E)
# --------------------------------------------------------------------------
W0 = 1.0         # Initial wealth
T = 30           # Time horizon in years
rf = 0.02        # Risk-free rate
mu = 0.07        # Risky asset mean return (arithmetic)
sigma = 0.16     # Risky asset volatility
theta = 3.37     # Agent's true coefficient of relative risk aversion
NUM_SIMS = 50000 # Number of Monte Carlo paths

# --------------------------------------------------------------------------
# 2. DEFINE STRATEGIES AND HELPER FUNCTIONS
# --------------------------------------------------------------------------

def calculate_optimal_allocation(mu, rf, theta, sigma):
    """Calculates the optimal equity allocation using the Merton formula."""
    return (mu - rf) / (theta * sigma**2)

def run_portfolio_simulation(w_risky, params):
    """
    Runs a Monte Carlo simulation for a given constant portfolio allocation.
    Returns the array of terminal wealth values.
    """
    W0, T, rf, mu, sigma, num_sims = params
    
    # Initialize wealth for all paths
    wealth_paths = np.full(num_sims, W0)
    
    # Pre-calculate the geometric mean component for log-normal returns
    drift = mu - (sigma**2 / 2)
    
    for year in range(T):
        # Generate random shocks for the current year for all paths
        Z = np.random.randn(num_sims)
        
        # Calculate risky asset returns for all paths (log-normal distribution)
        risky_returns = np.exp(drift + sigma * Z)
        
        # Calculate portfolio returns for all paths
        portfolio_returns = w_risky * risky_returns + (1 - w_risky) * (1 + rf)
        
        # Update wealth for all paths
        wealth_paths *= portfolio_returns
        
    return wealth_paths

def calculate_metrics(terminal_wealth, theta):
    """Calculates and returns key performance and welfare metrics."""
    # Define the CRRA utility function
    utility_func = lambda W: (W**(1 - theta)) / (1 - theta)
    
    # Calculate metrics
    median_wealth = np.median(terminal_wealth)
    mean_wealth = np.mean(terminal_wealth)
    expected_utility = np.mean(utility_func(terminal_wealth))
    
    # Calculate Certainty-Equivalent Wealth (CEW)
    cew = (expected_utility * (1 - theta))**(1 / (1 - theta))
    
    return {
        "Median Wealth": median_wealth,
        "Mean Wealth": mean_wealth,
        "CE Wealth": cew
    }

# --------------------------------------------------------------------------
# 3. RUN SIMULATIONS AND CALCULATE RESULTS
# --------------------------------------------------------------------------

# Set a random seed for reproducibility
np.random.seed(42)

# Define strategies
w_informed = calculate_optimal_allocation(mu, rf, theta, sigma)
w_uninformed = 0.35
params = (W0, T, rf, mu, sigma, NUM_SIMS)

# Run simulations
terminal_wealth_informed = run_portfolio_simulation(w_informed, params)
terminal_wealth_uninformed = run_portfolio_simulation(w_uninformed, params)

# Calculate metrics for both agents
metrics_informed = calculate_metrics(terminal_wealth_informed, theta)
metrics_uninformed = calculate_metrics(terminal_wealth_uninformed, theta)

# Calculate welfare loss
welfare_loss = (metrics_informed["CE Wealth"] - metrics_uninformed["CE Wealth"]) / metrics_informed["CE Wealth"]

# --------------------------------------------------------------------------
# 4. PRINT AND VERIFY THE RESULTS
# --------------------------------------------------------------------------
print("--- Verification of Wealth Drag Simulation (Table 4) ---")
print(f"\nInformed Agent's Optimal Equity Allocation: {w_informed:.1%}\n")

# Print table header
print(f"{'Outcome Metric':<25} {'Informed Agent (A)':>20} {'Uninformed Agent (B)':>22}")
print("-" * 70)

# Print results row by row
print(f"{'Equity Allocation':<25} {w_informed:>19.1%} {w_uninformed:>22.1%}")
print(f"{'Median Terminal Wealth':<25} ${metrics_informed['Median Wealth']:>18.2f} ${metrics_uninformed['Median Wealth']:>21.2f}")
print(f"{'Mean Terminal Wealth':<25} ${metrics_informed['Mean Wealth']:>18.2f} ${metrics_uninformed['Mean Wealth']:>21.2f}")
print("-" * 70)
print(f"{'Certainty-Equivalent Wealth':<25} ${metrics_informed['CE Wealth']:>18.2f} ${metrics_uninformed['CE Wealth']:>21.2f}")
print(f"{'Welfare Loss (vs. Informed)':<25} {'-':>20} {welfare_loss:>21.1%}")
print("-" * 70)