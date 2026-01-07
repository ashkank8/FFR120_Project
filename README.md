# FFR120_Project

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time  # For timing


def get_energy(grid, N, i, j):   # local energy of a site (i, j) based on neighbors

    S = grid[i, j]
    neighbors = grid[(i+1)%N, j] + grid[(i-1)%N, j] + \
                grid[i, (j+1)%N] + grid[i, (j-1)%N]
    return -1.0 * S * neighbors  # Coupling J=1.0

def run_kawasaki(N, T, steps, grid_initial=None):  #Simulates the Ising Model using Kawasaki Dynamics (Conserved Order Parameter).

    if grid_initial is None:
        # Initialize with exactly 50% spin +1 (Protein) and 50% spin -1 (Solvent)
        spins = np.array([1] * (N*N//2) + [-1] * (N*N//2))
        np.random.shuffle(spins)
        grid = spins.reshape(N, N)
    else:
        grid = grid_initial.copy()
    
    print(f"Starting Kawasaki simulation: N={N}, T={T}, steps={steps}")
    start_time = time.time()
    
    for sweep in range(steps):
        if sweep % (steps // 10) == 0: 
            print(f"  Sweep {sweep}/{steps} ({100*sweep//steps}%) - Elapsed: {time.time() - start_time:.1f}s")
        
        # Monte Carlo Sweep = N*N attempts
        for _ in range(N * N):
            i, j = np.random.randint(0, N, 2)
            
            
            # 0: Right, 1: Left, 2: Up, 3: Down
            direction = np.random.randint(0, 4)
            if direction == 0: ni, nj = (i + 1) % N, j
            elif direction == 1: ni, nj = (i - 1) % N, j
            elif direction == 2: ni, nj = i, (j + 1) % N
            else: ni, nj = i, (j - 1) % N
            
            # Only attempt swap if spins are different
            if grid[i, j] != grid[ni, nj]:
                # Calculate Energy change (dE) if swap
                E_initial = get_energy(grid, N, i, j) + get_energy(grid, N, ni, nj)
                
                # Temporarily swap to check new energy
                grid[i, j], grid[ni, nj] = grid[ni, nj], grid[i, j]  # Swap
                E_final = get_energy(grid, N, i, j) + get_energy(grid, N, ni, nj)
                
                dE = E_final - E_initial
                
                # Metropolis
                if dE > 0 and np.random.rand() >= np.exp(-dE / T):
                    grid[i, j], grid[ni, nj] = grid[ni, nj], grid[i, j]
                
    
    print(f"Simulation finished in {time.time() - start_time:.1f}s")
    return grid


def plot_figure1_evolution():   
    N = 100  
    T = 1.5  # Below critical temp (approx 2.27)
    steps_list = [0, 1000, 10000] 

    # Setup Initial Grid (Conserved 50/50)
    spins = np.array([1] * (N*N//2) + [-1] * (N*N//2))
    np.random.shuffle(spins)
    grid = spins.reshape(N, N)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    cmap = mcolors.ListedColormap(['#F1C40F', '#2980B9']) 

    snapshots = []
    titles = ["Homogeneous Mix", "Phase Separation (t=1000)", "Domain Growth (t=10000)"]

    snapshots.append(grid.copy())

    grid = run_kawasaki(N, T, steps=1000, grid_initial=grid)
    snapshots.append(grid.copy())

    grid = run_kawasaki(N, T, steps=9000, grid_initial=grid) 
    snapshots.append(grid.copy())

    for a, g, t in zip(ax, snapshots, titles):
        a.imshow(g, cmap=cmap)
        a.set_title(t, fontsize=14, fontweight='bold')
        a.axis('off')

    plt.tight_layout()
    plt.savefig('kawasaki_simulation.png', dpi=150) 
    plt.show() 



def plot_figure2_transition():
    print("Generating Figure 2: Transition (Energy vs T)...")
    N = 30 # Small lattice
    temperatures = np.linspace(1.0, 4.0, 15)
    energies = []
    
    for T in temperatures:
        # Initialize Random 50/50
        spins = np.array([1] * (N*N//2) + [-1] * (N*N//2))
        np.random.shuffle(spins)
        grid = spins.reshape(N, N)
        
        # Run to Equilibrium
        grid = run_kawasaki(N, T, steps=500, grid_initial=grid)
        
        # Calculate Final Energy
        E_total = 0
        for i in range(N):
            for j in range(N):
                S = grid[i, j]
                # Only sum right and down neighbors to avoid double counting
                neighbors = grid[(i+1)%N, j] + grid[i, (j+1)%N]
                E_total += -1 * S * neighbors
        
        avg_energy_per_spin = E_total / (N*N)
        energies.append(avg_energy_per_spin)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(temperatures, energies, 'o-', color='#2C3E50', linewidth=2)
    
    # Critical Temp
    Tc = 2.269
    ax.axvline(x=Tc, color='#E74C3C', linestyle='--', label=f'$T_c$')
    
    ax.set_xlabel('Temperature ($T$)', fontsize=14)
    ax.set_ylabel('Energy Density', fontsize=14)
    ax.set_title('Phase Transition', fontsize=16)
    
    # Adjusted positions to avoid overlap and place near data points
    min_idx = np.argmin(energies)
    max_idx = np.argmax(energies)
    ax.text(temperatures[min_idx], energies[min_idx] - 0.1, 'Phase Separated\n(Low Energy)', 
            color='#2980B9', fontweight='bold', ha='center', va='top')
    ax.text(temperatures[max_idx], energies[max_idx] + 0.1, 'Mixed\n(High Energy)', 
            color='#E74C3C', fontweight='bold', ha='center', va='bottom')
    
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('energyvstemperature.png', dpi=150)
    plt.show()


def plot_figure3_diagram():
    print("Generating Figure 3: Phase Diagram...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    T_range = np.linspace(0, 5, 100)
    J_boundary = T_range / 2.269
    
    ax.plot(T_range, J_boundary, color='black', linewidth=3, linestyle='--')
    ax.fill_between(T_range, J_boundary, 5, color='#D4E6F1', alpha=0.5) 
    ax.fill_between(T_range, 0, J_boundary, color='#FADBD8', alpha=0.5) 
    
    ax.set_xlabel('Temperature ($T$)', fontsize=14)
    ax.set_ylabel('Coupling Strength ($J$)', fontsize=14)
    ax.set_title('Phase Diagram', fontsize=16)
    

    
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 2.5)
    plt.tight_layout()
    plt.savefig('phasediagram.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    plot_figure1_evolution()
    plot_figure2_transition()
    plot_figure3_diagram()
