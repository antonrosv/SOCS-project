import numpy as np
import matplotlib.pyplot as plt

R = 1e-9  # Radius of the Brownian particle (motor head) [m].
eta = 1e-3  # Viscosity of the medium.
gamma_motor = 6 * np.pi * R * eta  # Drag coefficient of the motor head. 

# Cargo parameters
R_cargo = 50e-9  # Radius of the cargo [m] - typical vesicle size 50-100 nm
gamma_cargo = 6 * np.pi * R_cargo * eta  # Drag coefficient of the cargo.
gamma = gamma_motor + gamma_cargo  # Total drag coefficient

rho = 2e3  # Density of the particle [kg/m^3]
m = 4 * np.pi / 3 * rho * R ** 3  # Mass of the particle [kg].

tau = m / gamma  # Momentum relaxation time.

dt = 1e-12  # Time step [s].
duration = 10e-6  # Total time [s].



x0 = 0  # Initial position [m].
v0 = 0  # Initial velocity [m/s].

def spring_force(x):
    k = 0.0002
    y = x-8e-9
    return -y*k

def binding_force(x):
    kBT = 4.11e-21
    sigma = 1e-9/2
    xpos = 16e-9
    x = x- xpos
    U = 16*kBT
    return -U*x/(sigma**2)*np.exp(-x**2/ (2*sigma**2))

def cargo_load_force(R_cargo, eta):


    if R_cargo == 0:
        return 0.0
    
    gamma_cargo = 6 * np.pi * R_cargo * eta
    
    # Use a velocity scale that gives physically reasonable forces
    # For 50 nm cargo: gamma_cargo ~ 9.4e-10 N·s/m
    # To get ~0.5 pN load: need v ~ 0.5e-12 / 9.4e-10 ~ 0.5 mm/s
    # This represents effective velocity during stepping motion
    v_effective = 0.5e-3  # 0.5 mm/s = 500 μm/s
    
    F_load = -gamma_cargo * v_effective
    
    return F_load

#x_grid = np.linspace(-5*1e-9/2, 50*1e-9/2, 500)

#plt.plot(x_grid, binding_force(x_grid)+spring_force(x_grid))
#plt.show()


def evolution_viscous(x0, gamma, dt, duration, R_cargo=50e-9, eta=1e-3):
    """
    Function to generate the solution for the Langevin equation with 
    inertia and cargo load force.
    
    Parameters
    ==========
    x0 : Initial position of the oscillator [m].
    gamma : Friction coefficient of motor head [N*s/m].    
    dt : Time step for the numerical solution [s].
    duration : Total time for which the solution is computed [s].
    R_cargo : Radius of the cargo [m]. Default: 50e-9 (50 nm). Set to 0 for no cargo.
    eta : Viscosity of medium [Pa*s]. Default: 1e-3 (water).
    """
    
    # Calculate cargo load force (constant hindering force)
    F_load = cargo_load_force(R_cargo, eta) if R_cargo > 0 else 0
    
    kBT = 4.11e-21  # kB*T at room temperature [J].
    D = kBT / gamma  # Diffusion constant [m^2 / s] (only head, not cargo).
    
    # Coefficients for the finite difference solution.
    c_noise = np.sqrt(2 * D * dt)

    N = int(np.ceil(duration / dt))  # Number of time steps.

    x = np.zeros(N)
    rn = np.random.normal(0, 1, N - 1)
    
    x[0] = x0
    x_eq = 16e-9
    stable_time = 5e-9
    eps = 3e-10

    stable_steps_required = int(stable_time / dt)
    stable_counter = 0
    
    for i in range(N - 1):
        f = spring_force(x[i]) + binding_force(x[i]) + F_load
        x[i + 1] = x[i] + c_noise * rn[i] + f*dt/gamma
        if i % N*0.01 == 0:
            print ('max time elapsed')
            print (i/N)
        if abs(x[i+1] - x_eq) < eps:
            stable_counter += 1
        else:
            stable_counter = 0
        
        # Stop when condition met
        if stable_counter >= stable_steps_required:
            print(f"Stopped early at t = {i*dt:.3e} s.")
            return x[:i+2], i
    return x, i
