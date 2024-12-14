import numpy as np

#Defining Parameters for algorithm:
phi = 51.0486 # Latitude in Calgary,Alberta.
g0 = 9.811636 # measured in  m/s^2.
dg_dz = 3.086e-6 #g′ ≈ 0.3086 mGal/m where 1 Gal = 1 cm/s2, so in SI units, g′ ≈ (m/s2)/m.
rho_steel = 7800 # measured in  kg/m3.
d = 0.015 # density converted to m from cm (1.5cm).
μ_air =  1.827e-5 # measured in kg/(m⋅s).

#Defining Drag coefficient using stokes drag law for spherical objects:
r = d / 2 # radius of object.
volume = (4 / 3) * np.pi * r**3 # volume of sphere calculation.
mass = rho_steel *volume  # Mass of spherical object.
cD = 6 * np.pi * μ_air * r # Drag coeff.
cd_star = cD / mass     # Normalized Drag coeff.


#Euler method for freefall: Implementing numerical integration.
def ode_freefall_euler(g0, dg_dz, cd_star, H, dt):
    t, z, v = [0], [0], [0]
    while z[-1] < H:
        g = g0 - dg_dz * z[-1]
        drag = cd_star * v[-1]
        a = g - drag
        v_new = v[-1] + a * dt
        z_new = z[-1] + v[-1] * dt
        t_new = t[-1] + dt

        #Control on final step when z_new goes beyond H.
        if z_new > H:
            if abs(v[-1]) > 1e-6:
                dt_last = (H -z[-1]) / v[-1]
            else:
                dt_last = dt
            z_new = H
            t_new = t[-1] + dt_last
            v_new = v[-1] + a * dt_last

        #Updates values to the arrays.
        t.append(t_new)
        z.append(z_new)
        v.append(v_new)

        #Set condition made to hault simulation if velocity = zero ad expected height is obtained.
        if abs(v[-1]) < 1e-6 and z[-1] >= H:
            break
    
    return np.array(t), np.array(z), np.array(v)

#RK4 method of freefall: Implementing numerical integration.
def ode_freefall_rk4(g0, dg_dz, cd_star, H, dt):
    t, z, v = [0], [0], [0]
    while z[-1] < H:
        
        #Acceleration function to calculate net acceleration:
        def acceleration(z, v):
            g = g0 - dg_dz * z
            return g - cd_star * v
        
        #Rate of change of position function.
        def dz_dt(v):
            return v
        
        #Functions to ciompute intermediary steps  for gradients for velocity and position between k1 and k4.
        k1_v = acceleration(z[-1], v[-1]) * dt
        k1_z = dz_dt(v[-1]) * dt

        k2_v = acceleration(z[-1] + k1_z / 2, v[-1] + k1_v / 2) * dt
        k2_z = dz_dt(v[-1] + k1_v / 2) * dt

        k3_v = acceleration(z[-1] + k2_z / 2, v[-1] + k2_v / 2) * dt
        k3_z = dz_dt(v[-1] + k2_v / 2) * dt

        k4_v = acceleration(z[-1] + k3_z, v[-1] + k3_v) * dt
        k4_z = dz_dt(v[-1] + k3_v) * dt

        #Function to combine gradients to calculate final updates for  velocity and position.
        dv = (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6
        dz = (k1_z + 2 * k2_z + 2 * k3_z + k4_z) / 6

        v_new = v[-1] + dv
        z_new = z[-1] + dz
        t_new = t[-1] + dt

        #Control to final step if z_new exceeds H.
        if z_new > H:
            if abs(v[-1]) > 1e-6:
                dt_last = (H - z[-1]) / v[-1]
            else:
                dt_last = dt
            z_new = H
            t_new = t[-1] + dt_last
            v_new = v[-1] + acceleration(z[-1], v[-1]) * dt_last

        #Updates values to the arrays.
        t.append(t_new)
        z.append(z_new)
        v.append(v_new)

        #Set condition made to hault simulation if velocity = zero ad expected height is obtained.
        if abs(v[-1]) < 1e-6 and z[-1] >= H:
            break


    return np.array(t), np.array(z), np.array(v) 