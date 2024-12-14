# Test case 1.
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ODE_Case_Solver import ode_freefall_euler, ode_freefall_rk4

def compute_relative_error(true_value, computed_values):
    return np.abs(computed_values - true_value) / true_value

# Parameter definition and coefficient calculation.
#Parameters:
phi = 51.0486 # Latitude in Calgary,Alberta.
g0 = 9.811636 # measured in  m/s^2.
dg_dz = 3.086*10**-6 #g′ ≈ 0.3086 mGal/m where 1 Gal = 1 cm/s2, so in SI units, g′ ≈ (m/s2)/m.
rho_steel = 7800 # measured in  kg/m3.
d = 0.015 # density converted to m from cm (1.5cm).
μ_air =  1.827*10**-5 # measured in kg/(m⋅s).

# Drag coefficient:
r = d / 2
volume = (4 / 3) * np.pi * r**3
mass = rho_steel *volume
cD = 6 * np.pi * μ_air * r
cd_star = cD / mass

# heights in meters
heights = [10, 20, 40]
dt_values = np.logspace(-4, -1, 10)
dt_ref = 1e-5

euler_times = {H: [] for H in heights}
rk4_times = {H: [] for H in heights}
ref_times = {}


for H in heights:
    _, _, v_euler = ode_freefall_euler(g0, dg_dz, cd_star, H, dt_ref)
    _, _, v_rk4 = ode_freefall_rk4(g0, dg_dz, cd_star, H, dt_ref)
    ref_times[H] = v_rk4[-1]

for H in heights:
    for dt in dt_values:
        _, _, v_euler = ode_freefall_euler(g0, dg_dz, cd_star, H, dt)
        _, _, v_rk4 = ode_freefall_rk4(g0, dg_dz, cd_star, H, dt)
        euler_times[H].append(v_euler[-1])
        rk4_times[H].append(v_rk4[-1])

plt.figure(figsize=(12, 6))
for H in heights:
    plt.plot(dt_values, euler_times[H], label=f"Euler H={H} m", marker='o', linestyle='--')
    plt.plot(dt_values, rk4_times[H], label=f"RK4 H={H} m", marker='x', linestyle='-')

plt.xscale('log')
plt.xlabel("Time Step  Δt (s)")
plt.ylabel("Total Drop Time t* (s)")
plt.title("Simulatiofn Time vs. Time step for Euler and RK4 Methods")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()

plt.figure(figsize=(12, 6))
for H in heights:
    euler_errors = compute_relative_error(ref_times[H], np.array(euler_times[H]))
    rk4_errors = compute_relative_error(ref_times[H], np.array(rk4_times[H]))
    plt.plot(dt_values, euler_errors[H], label=f"Euler Error H={H} m", marker='o', linestyle='--')
    plt.plot(dt_values, rk4_times[H], label=f"RK4 Error H={H} m", marker='x', linestyle='-')

plt.xscale('log')
plt.yscale('log')
plt.xlabel("Time Step  Δt (s)")
plt.ylabel("Relative Error")
plt.title("Relative Error vs. Time step Δt")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()