import numpy as np
import matplotlib.pyplot as plt

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

x_grid = np.linspace(-5*1e-9/2, 50*1e-9/2, 500)

plt.plot(x_grid, binding_force(x_grid)+spring_force(x_grid))
plt.show()

def evolution_viscous(x0, gamma, dt, duration):
    """
    Function to generate the solution for the Langevin equation with 
    inertia.
    
    Parameters
    ==========
    x0 : Initial position of the oscillator [m].
    gamma : Friction coefficient [N*s/m].    
    dt : Time step for the numerical solution [s].
    duration : Total time for which the solution is computed [s].
    """
    
    kBT = 4.11e-21  # kB*T at room temperature [J].
    
    D = kBT / gamma  # Diffusion constant [m^2 / s].
    
    # Coefficients for the finite difference solution.
    c_noise = np.sqrt(2 * D * dt)

    N = int(np.ceil(duration / dt))  # Number of time steps.

    x = np.zeros(N)
    rn = np.random.normal(0, 1, N - 1)
    
    x[0] = x0

    for i in range(N - 1):
        f = spring_force(x[i]) + binding_force(x[i])
        x[i + 1] = x[i] + c_noise * rn[i] + f*dt/gamma

    return x, D

# Simulation for a colloidal particle in water at room temperature.

R = 1e-9  # Radius of the Brownian particle [m].
eta = 1e-3  # Viscosity of the medium.
gamma = 6 * np.pi * R * eta  # Drag coefficient of the medium. 
rho = 2.e+3  # Density of the particle [kg/m^3]
m = 4 * np.pi / 3 * rho * R ** 3  # Mass of the particle [kg].


tau = m / gamma  # Momentum relaxation time.

dt = 1e-12  # Time step [s].
duration = 20e-6  # Total time [s].



x0 = 0  # Initial position [m].
v0 = 0  # Initial velocity [m/s].

print(f'tau={tau:.3e} s.') 

x_visc, D_v = evolution_viscous(x0, gamma, dt, duration)

t = dt * np.arange(int(np.ceil(duration / dt)))

plt.plot(t / tau, x_visc, '-', color='b', linewidth=0.5, label='viscous')


plt.legend()

plt.title('Trajectories')

plt.xlabel('t (tau)')
plt.ylabel('x (m)')

plt.show()
