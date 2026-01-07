import numpy as np
import matplotlib.pyplot as plt

R = 1e-9  # Radius of the Brownian particle (motor head) [m].
eta = 1e-3  # Viscosity of the medium.
gamma_motor = 6 * np.pi * R * eta  # Drag coefficient of the motor head. 

# Cargo parameters
R_cargo = 50e-9  # Radius of the cargo [m] - typical vesicle size 50-100 nm
gamma_cargo = 6 * np.pi * R_cargo * eta  # Drag coefficient of the cargo.
gamma = gamma_motor
rho = 2e3  # Density of the particle [kg/m^3]
m = 4 * np.pi / 3 * rho * R ** 3  # Mass of the particle [kg].

tau = m / gamma  # Momentum relaxation time.

dt = 1e-12  # Time step [s].
duration = 5e-6  # Total time [s].



x0 = 0  # Initial position [m].
v0 = 0  # Initial velocity [m/s].

def spring_force(x):
    k = 0.0002
    y = x-8e-9
    return -y*k

def binding_force(x, kBT=4.11e-21):
    """
    kBT: Thermal energy [J]. Default corresponds to ~298 K.
    """
    sigma = 1e-9/2
    xpos = 16e-9
    x = x- xpos
    x_back = x + 3*xpos/2
    kBT = 4.11e-21
    U = 16*kBT
    return -U*x/(sigma**2)*np.exp(-x**2/ (2*sigma**2)) + -U*x_back/(sigma**2)*np.exp(-x_back**2/ (2*sigma**2))


def cargo_load_force(R_cargo, eta):
    return 0.5e-12



def evolution_viscous(x0, gamma, dt, duration, R_cargo=50e-9, eta=1e-3, T=298.0):
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
    T : Temperature [K]. Default: 298.0 K (room temperature).
    """
    
    # Calculate cargo load force (constant hindering force)
    F_load = cargo_load_force(R_cargo, eta) if R_cargo > 0 else 0
    
    kB = 1.38e-23  # Boltzmann constant [J/K]
    kBT = kB * T  # Thermal energy [J]
    D = kBT / gamma  # Diffusion constant [m^2 / s] (only head, not cargo).
    
    # Coefficients for the finite difference solution.
    c_noise = np.sqrt(2 * D * dt)

    N = int(np.ceil(duration / dt))  # Number of time steps.

    x = np.zeros(N)
    rn = np.random.normal(0, 1, N - 1)
    
    x[0] = x0
    x_eq = 16e-9
    stable_time = 5e-6
    eps = 4e-10

    stable_steps_required = int(stable_time / dt)
    #stable_steps_required = 10
    stable_counter = 0
    
    for i in range(N - 1):
        f = spring_force(x[i]) + binding_force(x[i], kBT) + F_load
        x[i + 1] = x[i] + c_noise * rn[i] + f*dt/gamma
        if abs(x[i+1] - x_eq) < eps:
            stable_counter += 1
        else:
            stable_counter = 0
        
        # Stop when condition met
        #if stable_counter >= stable_steps_required:
            #print(f"Stopped early at t = {i*dt:.3e} s.")
            #return x[:i+2], i
    print('Max time elapsed')
    return x, i
plt.figure(figsize=(7, 7), facecolor='#DBF0F2')

for j in range(10):
    T = 600
    x , i = evolution_viscous(0, gamma, dt, duration, R_cargo, eta, T)
    time = np.linspace(0, i, i+2)
    plt.plot(time*dt*10e2, x, color = 'r', alpha = 0.5)
plt.xlabel('Time (ms)')
plt.ylabel('x (m)')
plt.title('T = 600K')

plt.show()

plt.figure(figsize=(7,7), facecolor='#DBF0F2')
for j in range(10):
    T = 300
    x , i = evolution_viscous(0, gamma, dt, duration, R_cargo, eta, T)
    time = np.linspace(0, i, i+2)
    plt.plot(time*dt*10e2, x, color = 'g', alpha = 0.5)
plt.xlabel('Time (ms)')
plt.ylabel('x (m)')
plt.title('T = 300K')


plt.show()

plt.figure(figsize=(7, 7), facecolor='#DBF0F2')
for j in range(10):
    T = 400
    x , i = evolution_viscous(0, gamma, dt, duration, R_cargo, eta, T)
    time = np.linspace(0, i, i+2)
    plt.plot(time*dt*10e2, x, color = 'b', alpha = 0.5)
plt.xlabel('Time (ms)')
plt.ylabel('x (m)')
plt.title('T = 400K')

plt.show()



