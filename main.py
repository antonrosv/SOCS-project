import numpy as np
import matplotlib.pyplot as plt
from forces import evolution_viscous

dt = 1e-12
duration = 10e-6
R = 1e-9  # Radius of the Brownian particle [m].
eta = 1e-3  # Viscosity of the medium.
gamma_motor = 6 * np.pi * R * eta  # Drag coefficient of the motor head.

# Cargo parameters
R_cargo = 50e-9  # Radius of the cargo [m] - typical vesicle size
gamma_cargo = 6 * np.pi * R_cargo * eta  # Drag coefficient of the cargo.
gamma = gamma_motor + gamma_cargo  # Total drag coefficient 


class Head:
    def __init__(self, nucleotide_state, attached, start_position):
        self.nucleotide_state = nucleotide_state  # 'ADP', 'ATP', or 'nucleotide-free'
        self.attached = attached  # True if attached to microtubule, False otherwise
        self.position = start_position * 8e-9
    def bind_ATP(self):
        if self.nucleotide_state == 'free':
            self.nucleotide_state = 'ATP'
    def release_ADP(self):
        if self.nucleotide_state == 'ADP':
            self.nucleotide_state = 'free'
    def ATP_to_ADP(self):
        if self.nucleotide_state == 'ATP':
            self.nucleotide_state = 'ADP'
            self.attached = False
    def move(self):
        iterations = evolution_viscous(0, gamma_motor, dt, duration, R_cargo, eta)[1]
        self.position += 16e-9
        return iterations
    def attach(self):
        self.attached = True


def evolution(head_front, head_back, ATP_concentration):
    time = 0
    while ATP_concentration >= 1:
        print (ATP_concentration)
        if head_front.nucleotide_state == 'free' and head_back.nucleotide_state == 'ADP':
            running = True
            k_ATP = 100
            wait_time = np.random.exponential(1/k_ATP)
            time += wait_time
            head_front.bind_ATP()

            iterations = head_back.move()
            time += dt*iterations
            head_front, head_back = head_back, head_front

            head_front.release_ADP()

            head_back.ATP_to_ADP()

            head_front.attach()

            ATP_concentration -= 1
    return time, head_front.position

head_front = Head('free', True, 1)
head_back = Head('ADP', False, 0)


#print(f'tau={tau:.3e} s.') 

time, position = evolution(head_front, head_back, 50)
print (time, position)
print (position*1e9/time)

# Optional: Uncomment below to test individual Brownian trajectory
#x0 = 0
#x_visc, D_v = evolution_viscous(x0, gamma_motor, dt, duration, R_cargo, eta)
#t = dt * np.arange(int(np.ceil(duration / dt)))
#plt.plot(x_visc, '-', color='b', linewidth=0.5)
#plt.title('Trajectories')
#plt.xlabel('t')
#plt.ylabel('x (m)')
#plt.show()
