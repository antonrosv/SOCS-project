from scipy import constants

viscosity = 1                   
kB = constants.Boltzmann        
d = 8.1e-9                      #Distance between nodes on microtubule
k = 0.0002                      #Spring constant for neck liner
C_d = 1                         #Drag coefficient for bead
binding_time = 1                #Time a particle has to be in the potential well for it to be considered bound
bead_size = 2e-9                

import numpy as np
import matplotlib.pyplot as plt

def binding_force(x, kBT=4.11e-21):
    """
    kBT: Thermal energy [J]. Default corresponds to ~298 K.
    """
    sigma = 1e-9/2
    xpos = 16.2e-9
    x = x- xpos
    x_back = x + xpos + xpos/2
    kBT = 4.11e-21
    U = 16*kBT
    return -U*x/(sigma**2)*np.exp(-x**2/ (2*sigma**2)) + -U*x_back/(sigma**2)*np.exp(-x_back**2/ (2*sigma**2))


x_grid = np.linspace((-50*1e-9)/2, (50*1e-9/2), 5000)

plt.plot(x_grid, binding_force(x_grid))
plt.show()

for i in range(5):
    T = 200
    x , i = evolution_viscous(0, gamma, dt, duration, R_cargo, eta, T)

    plt.plot(x, color = 'r', alpha = '0.5')
plt.show
for i in range(5):
    T = 300
    x , i = evolution_viscous(0, gamma, dt, duration, R_cargo, eta, T)

    plt.plot(x, color = 'g')

for i in range(5):
    T = 400
    x , i = evolution_viscous(0, gamma, dt, duration, R_cargo, eta, T)

    plt.plot(x, color = 'b')

plt.show()