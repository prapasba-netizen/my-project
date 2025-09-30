import numpy as np
import matplotlib.pyplot as plt

# Parameters from your paper
beta = 0.96
r = 0.04
theta = 5  # risk aversion
rho = 2/3  # inverse EIS = 1/1.5
W = 1.0    # initial wealth

def u_A(c):
    """Admissible CRRA utility"""
    return (c**(1-theta)) / (1-theta)

def u_B(c):
    """Inadmissible normalized CRRA utility"""
    return (c**(1-theta) - 1) / (1-theta)

def lifetime_utility_A(s):
    """Lifetime utility for admissible representation - REAL VERSION"""
    c1 = W - s
    c2 = s * (1 + r)
    
    # For theta > 1, u_A(c) is negative, so we need to handle signs carefully
    u1 = u_A(c1)  # negative
    u2 = u_A(c2)  # negative
    
    # Epstein-Zin aggregator - using absolute values and restoring sign
    term1 = (1-beta) * (abs(u1))**(1-rho)
    term2 = beta * (abs(u2))**(1-rho)  # Simplified since (1-rho)/(1-theta) = 1 when rho=theta
    
    V1 = -(term1 + term2)**(1/(1-rho))  # Negative result
    return V1

def lifetime_utility_B(s):
    """Lifetime utility for inadmissible representation - REAL VERSION"""
    c1 = W - s
    c2 = s * (1 + r)
    
    u1 = u_B(c1)  # negative for c<1
    u2 = u_B(c2)  # negative for c<1
    
    # Epstein-Zin aggregator
    term1 = (1-beta) * (abs(u1))**(1-rho)
    term2 = beta * (abs(u2))**(1-rho)
    
    V1 = -(term1 + term2)**(1/(1-rho))
    return V1

# Generate savings rates
savings_rates = np.linspace(0.75, 0.85, 100)

# Calculate utilities
utility_A = [lifetime_utility_A(s) for s in savings_rates]
utility_B = [lifetime_utility_B(s) for s in savings_rates]

# Find optimal savings rates (we want to MAXIMIZE utility, so find least negative)
opt_s_A = savings_rates[np.argmax(utility_A)]  # argmax finds least negative
opt_s_B = savings_rates[np.argmax(utility_B)]

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot utility curves (they will be negative)
ax.plot(savings_rates, utility_A, label='Admissible Representation ($u_A$)', linewidth=2.5, color='blue')
ax.plot(savings_rates, utility_B, label='Inadmissible Representation ($u_B$)', linewidth=2.5, color='red')

# Add vertical lines at optimal points
ax.axvline(x=opt_s_A, color='blue', linestyle='--', alpha=0.7, 
           label=f'Optimal for $u_A$: {opt_s_A:.3f}')
ax.axvline(x=opt_s_B, color='red', linestyle='--', alpha=0.7,
           label=f'Optimal for $u_B$: {opt_s_B:.3f}')

# Add markers at optimal points
ax.plot(opt_s_A, max(utility_A), 'bo', markersize=8)
ax.plot(opt_s_B, max(utility_B), 'ro', markersize=8)

# Labels and formatting
ax.set_xlabel('Savings Rate (s/W)', fontsize=12, fontweight='bold')
ax.set_ylabel('Lifetime Utility $V_1$ (Negative)', fontsize=12, fontweight='bold')
ax.set_title('Sensitivity of Lifetime Utility to Savings Rate Under Different Utility Representations', 
             fontsize=13, fontweight='bold', pad=20)

# Add grid
ax.grid(True, alpha=0.3)

# Legend
ax.legend(frameon=True, fancybox=True, framealpha=0.9, 
          loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2)

# Adjust layout to prevent legend cutoff
plt.tight_layout()

# Save as high-quality PNG for academic publication
plt.savefig('savings_sensitivity.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')

plt.show()

print(f"Optimal savings rate for admissible representation: {opt_s_A:.3f}")
print(f"Optimal savings rate for inadmissible representation: {opt_s_B:.3f}")
print(f"Difference: {abs(opt_s_A - opt_s_B):.3f}")

# Print some sample values for verification
print(f"\nVerification at optimal points:")
print(f"u_A at s={opt_s_A:.3f}: {utility_A[np.argmax(utility_A)]:.6f}")
print(f"u_B at s={opt_s_B:.3f}: {utility_B[np.argmax(utility_B)]:.6f}")