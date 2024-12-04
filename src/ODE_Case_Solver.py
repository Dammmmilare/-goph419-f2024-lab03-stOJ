import numpy as np

#Parameters:

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
cD = 6 * np.pi * μ_air 

def ode_freefall_euler(g0, dg_dz, cd_star, H, dt):
    t, z, v = [0], [0], [0]
