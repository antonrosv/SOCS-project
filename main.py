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
        # Start from current position, not zero
        x_values, iterations = evolution_viscous(self.position, gamma_motor, dt, duration, R_cargo, eta)
        # Update position to the final value from the trajectory
        self.position = x_values[-1]
        return x_values, iterations
    def attach(self):
        self.attached = True


def evolution(head_front, head_back, ATP_concentration, downsample_factor=1000):
    """
    downsample_factor: Only save every Nth point to reduce memory usage.
    With dt=1e-12 and downsample_factor=1000, we save every 1e-9 s.
    """
    time = 0
    all_x_values = []  # Store all x values (overall motor/cargo position)
    all_times = []  # Store corresponding time points
    
    # Track the overall forward position of the motor
    # This represents the cargo/motor center-of-mass that moves forward
    overall_position = head_front.position
    
    while ATP_concentration >= 1:
        print (ATP_concentration)
        if head_front.nucleotide_state == 'free' and head_back.nucleotide_state == 'ADP':
            running = True
            k_ATP = 100
            wait_time = np.random.exponential(1/k_ATP)
            time += wait_time
            head_front.bind_ATP()

            x_values, iterations = head_back.move()
            
            # Downsample to reduce memory usage
            x_values_downsampled = x_values[::downsample_factor]
            # Create time array for this step (downsampled)
            step_times = time + dt * np.arange(len(x_values))[::downsample_factor]
            
            # For kinesin, each successful step moves forward by 8nm (microtubule periodicity)
            # Show the forward progress: local Brownian search + overall forward stepping
            # The local motion is relative to the starting position
            relative_motion = x_values_downsampled - x_values[0]
            # Add overall position to show cumulative forward movement
            forward_trajectory = overall_position + relative_motion
            
            all_x_values.extend(forward_trajectory)
            all_times.extend(step_times)
            
            # After step completion, motor moves forward by 8nm (typical kinesin step size)
            step_size = 8e-9  # 8 nm forward step
            overall_position += step_size
            
            time += dt*iterations
            head_front, head_back = head_back, head_front

            head_front.release_ADP()

            head_back.ATP_to_ADP()

            head_front.attach()

            ATP_concentration -= 1
    return time, head_front.position, np.array(all_x_values), np.array(all_times)

head_front = Head('free', True, 1)
head_back = Head('ADP', False, 0)


#print(f'tau={tau:.3e} s.') 

time, position, x_values, t_values = evolution(head_front, head_back, 50)
print (time, position)
print (position*1e9/time)

# Plot x values against time
plt.figure(figsize=(10, 6))
plt.plot(t_values, x_values, '-', color='b', linewidth=0.5, label='Position trajectory')
plt.title('Kinesin Position vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# Optional: Uncomment below to test individual Brownian trajectory
#x0 = 0
#x_visc, iterations = evolution_viscous(x0, gamma_motor, dt, duration, R_cargo, eta)
#t = dt * np.arange(len(x_visc))
#plt.plot(t, x_visc, '-', color='b', linewidth=0.5)
#plt.title('Trajectories')
#plt.xlabel('t (s)')
#plt.ylabel('x (m)')
#plt.show()
