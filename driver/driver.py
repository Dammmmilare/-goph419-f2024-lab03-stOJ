#Test Case 3.
import  numpy as np 
import matplotlib.pyplot as plt
from src.ODE_Case_Solver import ode_freefall_euler, ode_freefall_rk4

def compute_sensitivity(solver, g0, dg_dz, cd_star, H, dt, alpha):
    
    #Baseline drop time
    _, _, v_base = solver(g0, dg_dz, cd_star, H, dt)
    t_base = v_base[-1]

    #perturb g0
    g0_perturbed = g0 * (1 + alpha)
    _, _, v_g0 = solver(g0_perturbed, dg_dz, cd_star, H, dt)
    delta_t_g0 = v_g0[-1] - t_base

    #peturb dg_dz
    dg_dz_perturbed = dg_dz * (1 + alpha)
    _, _, v_dgdz = solver(g0, dg_dz_perturbed, cd_star, H, dt)
    delta_t_dgdz = v_dgdz[-1] - t_base

    #peturb cd_star
    cd_star_perturbed = cd_star * (1 + alpha)
    _, _, v_cd = solver(g0, dg_dz, cd_star_perturbed, H, dt)
    delta_t_cd = v_cd[-1] - t_base

    return delta_t_g0, delta_t_dgdz, delta_t_cd

# Parameters
phi = 51.0486 # Latitude in Calgary,Alberta.
g0 = 9.811636 # measured in  m/s^2.
dg_dz = 3.086*10-6 #g′ ≈ 0.3086 mGal/m where 1 Gal = 1 cm/s2, so in SI units, g′ ≈ (m/s2)/m.
rho_steel = 7800 # measured in  kg/m3.
d = 0.015 # density converted to m from cm (1.5cm).
μ_air =  1.827*10-5 # measured in kg/(m⋅s).

# Drag coefficient:
r = d / 2
volume = (4 / 3) * np.pi * r**3
mass = rho_steel *volume
cD = 6 * np.pi * μ_air * r
cd_star = cD / mass
alpha =  1*10^-2
dt = 1e-4
heights = [10, 20, 40]

#Sensitive results
euler_sensitivities = {H: {} for H in heights}
rk4_sensitivities = {H: {} for H in heights}

#Computing sensitivities for each height
for H in heights:
    euler_sensitivities[H]['g0'], euler_sensitivities[H]['dgdz'], euler_sensitivities[H]['cd_star'] = compute_sensitivity(
        ode_freefall_euler, g0, dg_dz, cd_star, H, dt, alpha
    )
    rk4_sensitivities[H]['g0'], rk4_sensitivities[H]['dgdz'], rk4_sensitivities[H]['cd_star'] = compute_sensitivity(
        ode_freefall_rk4, g0, dg_dz, cd_star, H, dt, alpha
    )

#Plot results
labels = ['g0', 'dg_dz', 'cd_star']
for H in heights:
    plt.figure(figsize=(10, 5))
    euler_values = [euler_sensitivities[H][label] for label in labels]
    rk4_values = [rk4_sensitivities[H][label] for label in labels]

    plt.bar(labels, euler_values, alpha=0.6, label="Euler")
    plt.bar(labels, rk4_values, alpha=0.6, label="rk4")
    plt.title(f"Sensitivity of Drop Time for H = {H} m")
    plt.ylabel("Change in Drop Time (s)")
    plt.legend()
    plt.grid(axis='', linestyle='', linewidth=0.5)
    plt.show()