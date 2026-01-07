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
    def __init__(self, nucleotide_state, attached, start_position, T=298.0):
        self.nucleotide_state = nucleotide_state  # 'ADP', 'ATP', or 'nucleotide-free'
        self.attached = attached  # True if attached to microtubule, False otherwise
        self.position = start_position * 8.1e-9
        self.T = T  # Temperature [K]
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
        x_values, iterations = evolution_viscous(0, gamma_motor, dt, duration, R_cargo, eta, self.T)
        self.position += 16.2e-9
        return x_values, iterations
    def attach(self):
        self.attached = True


def evolution(head_front, head_back, ATP_concentration, downsample_factor=1000):
    """
    downsample_factor: Only save every Nth point to reduce memory usage.
    With dt=1e-12 and downsample_factor=1000, we save every 1e-9 s.
    """
    time = 0
    all_x_values = []  # Store cargo/motor position (shows standing still then stepping)
    all_times = []  # Store corresponding time points
    search_times = []  # Store the time for each Brownian search phase
    
    # Track the cargo/motor position - this stays relatively constant during search,
    # then jumps forward when step completes
    cargo_position = head_front.position
    
    # Get temperature from head (for cargo fluctuations)
    T = head_front.T
    kB = 1.38e-23  # Boltzmann constant [J/K]
    kBT = kB * T
    
    while ATP_concentration >= 1:
        print (ATP_concentration)
        if head_front.nucleotide_state == 'free' and head_back.nucleotide_state == 'ADP':
            running = True
            k_ATP = 100
            rng_atp = np.random.default_rng(ATP_concentration)
            wait_time = rng_atp.exponential(1/k_ATP)
            time += wait_time
            head_front.bind_ATP()

            # During this phase, head searches for binding site
            # Cargo position stays relatively constant (small fluctuations)
            x_values, iterations = head_back.move()
            
            # Track the search time (Brownian motion phase)
            search_time = dt * iterations
            search_times.append(search_time)
            
            # Downsample to reduce memory usage
            x_values_downsampled = x_values[::downsample_factor]
            # Create time array for this step (downsampled)
            step_times = time + dt * np.arange(len(x_values))[::downsample_factor]
            
            # During search phase: cargo stays at current position with small fluctuations
            # The head searches locally, but cargo (being much larger) doesn't move much
            # Small fluctuations represent thermal motion of cargo (temperature-dependent)
            cargo_fluctuation_scale = np.sqrt(kBT / (gamma_cargo * dt * downsample_factor)) * 1e-10
            small_fluctuations = np.random.normal(0, cargo_fluctuation_scale, len(x_values_downsampled))
            search_trajectory = cargo_position + small_fluctuations
            
            all_x_values.extend(search_trajectory)
            all_times.extend(step_times)
            
            # After step completion: cargo jumps forward by 8nm (typical kinesin step size)
            step_size = 8.1e-9  # 8 nm forward step
            cargo_position += step_size
            
            # Add a point showing the jump (instantaneous step)
            all_x_values.append(cargo_position)
            all_times.append(time + dt*iterations)
            
            time += dt*iterations
            head_front, head_back = head_back, head_front

            head_front.release_ADP()

            head_back.ATP_to_ADP()

            head_front.attach()

            ATP_concentration -= 1
    
    # Calculate statistics
    avg_search_time = np.mean(search_times)
    std_search_time = np.std(search_times)
    
    # Return cargo_position (cumulative forward movement), not head_front.position
    return time, cargo_position, np.array(all_x_values), np.array(all_times), avg_search_time, std_search_time

# Test different temperatures
temperatures = [200, 300, 400]  # K
colors = ['b', 'r', 'g']
labels = ['T = 200 K', 'T = 300 K', 'T = 400 K']

plt.figure(figsize=(12, 7), facecolor='#DBF0F2')

for T, color, label in zip(temperatures, colors, labels):
    print(f"\n{'='*60}")
    print(f"Running simulation at T = {T} K...")
    # Create new heads for each temperature
    head_front = Head('free', True, 1, T=T)
    head_back = Head('ADP', False, 0, T=T)
    
    time, position, x_values, t_values, avg_search_time, std_search_time = evolution(head_front, head_back, 250)
    
    print(f"\nResults for T = {T} K:")
    print(f"  Final position: {position*1e9:.2f} nm")
    print(f"  Total time: {time:.3e} s")
    print(f"  Average velocity: {position*1e9/time:.2f} nm/s")
    print(f"  Average Brownian search time per step: {avg_search_time:.3e} s ({avg_search_time*1e9:.2f} ns)")
    print(f"  Standard deviation: {std_search_time:.3e} s ({std_search_time*1e9:.2f} ns)")
    print(f"  Number of steps: 100")
    
    # Plot trajectory for this temperature
    plt.plot(t_values, x_values*1e9, '-', color=color, linewidth=0.5, label=label, alpha=0.7)

plt.title('Kinesin Position vs Time', fontsize=16)
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel('Position (nm)', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(
    fontsize=12,          # increase legend text size
    frameon=True,         # make sure legend box is visible
    facecolor='#DBF0F2',  # background color
    edgecolor='black'     # optional: legend border color
)
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
