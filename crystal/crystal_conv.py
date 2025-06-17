# --- convergence_test_TiN.py ---

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ase.io import read
from gpaw import GPAW, PW, FermiDirac

# --- Global Parameters ---
# Materials to test. Start with TiN, others commented out for now.
materials = ['TiN']
# materials = ['ZrN', 'VN', 'NbN', 'TiN', 'ScN'] # Uncomment and use for multiple metals later

# Lattice constant scaling factors for volume scan
scales = [round(0.90 + 0.02 * i, 3) for i in range(5)] + [1.0] + [1.02]
scales.sort() # Ensure scales are in increasing order

# Fixed k-point mesh for both convergence tests
K_POINTS_FIXED = (5, 5, 5)

# Smearing width for Fermi-Dirac occupations
SMEARING_WIDTH = 0.05

# Fixed Ecut value to use for the Lattice Constant scan
# This should be a value you expect to be converged or high enough
FIXED_ECUT_FOR_LC_SCAN = 500 # eV

# Range of Ecut values to test for Ecut convergence
ECUT_RANGE_FOR_CONVERGENCE = list(range(350, 601, 50)) + [650, 700]
ECUT_RANGE_FOR_CONVERGENCE.sort()

print("--- Starting Convergence Tests for Selected Metals ---")
print(f"  Fixed K-points for all tests: {K_POINTS_FIXED}")
print(f"  Smearing Width: {SMEARING_WIDTH}")

# --- Ecut Off Convergence (for each selected metal) ---
print("\n--- Running Ecut Off Convergence Tests ---")

# Store results for plotting
all_ecut_results = {}

for name in materials:
    print(f"\n  Processing {name} for Ecut Off convergence...")
    
    # Load the initial atoms object for this material
    # We use the original lattice constant for Ecut convergence
    try:
        atoms = read(f"{name}.cif") 
    except FileNotFoundError:
        print(f"Error: {name}.cif not found. Please ensure the .cif file is in the script's directory.")
        continue # Skip to the next metal if file is missing
    
    ecut_energies = []

    for ecut_val in ECUT_RANGE_FOR_CONVERGENCE:
        print(f"    Calculating for Ecut = {ecut_val} eV...")
        
        # Create a new GPAW calculator with the current Ecut
        calc = GPAW(mode=PW(ecut_val),
                    xc='PBE',
                    kpts={'size': K_POINTS_FIXED},
                    occupations=FermiDirac(SMEARING_WIDTH),
                    txt=None) # Suppress GPAW output to terminal for each step
        
        atoms_copy = atoms.copy() # Work on a copy to ensure clean state
        atoms_copy.set_calculator(calc)
        
        energy = atoms_copy.get_potential_energy()
        ecut_energies.append(energy)
        
        print(f"      Ecut {ecut_val} eV: Energy = {energy:.4f} eV")
        
        # Clean up calculator
        if hasattr(calc, 'destroy'):
            calc.destroy()
    
    all_ecut_results[name] = ecut_energies
    print(f"  Ecut Off Convergence for {name} complete.")

# --- Plotting Ecut Off Convergence for each metal ---
for name, energies in all_ecut_results.items():
    plt.figure(figsize=(10, 7))
    plt.plot(ECUT_RANGE_FOR_CONVERGENCE, energies, marker='o', linestyle='-', color='blue')

    plt.xlabel('Cutoff Energy (eV)', fontsize=12)
    plt.ylabel('Total Energy (eV)', fontsize=12)
    plt.title(f'GPAW PBE: Energy vs. Cutoff Energy Convergence for {name}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    print(f"Plot for PBE Ecut Off Convergence for {name} displayed.")


# --- Lattice Constant Convergence (for each selected metal) ---
print("\n--- Running Lattice Constant Convergence Tests ---")
print(f"  Fixed Ecut: {FIXED_ECUT_FOR_LC_SCAN} eV")

# Store results for plotting
all_lc_scan_data = {}
lc_values_template = [] # To store the actual lattice constant values (will be similar for all materials)

for name in materials:
    print(f"\n  Processing {name} for lattice constant scan...")
    
    try:
        atoms = read(f"{name}.cif") 
    except FileNotFoundError:
        print(f"Error: {name}.cif not found. Skipping {name} for LC scan.")
        continue
        
    original_cell = atoms.get_cell()
    
    current_energies = []
    current_lcs = [] # Store actual LC values for this material/run

    for scale in scales:
        atoms_scaled = atoms.copy() # Work on a copy for each scale
        atoms_scaled.set_cell(original_cell * scale, scale_atoms=True)
        
        # Define GPAW calculator for PBE
        calc = GPAW(mode=PW(FIXED_ECUT_FOR_LC_SCAN),
                    xc='PBE',
                    kpts={'size': K_POINTS_FIXED},
                    occupations=FermiDirac(SMEARING_WIDTH),
                    txt=None) # Suppress GPAW output to terminal for each step
        
        atoms_scaled.set_calculator(calc)
        
        energy = atoms_scaled.get_potential_energy()
        a = atoms_scaled.get_cell()[0, 0] # Assuming cubic cell, get 'a' parameter
        
        current_lcs.append(a)
        current_energies.append(energy)
        
        print(f"    {name} at scale {scale:.3f} (LC: {a:.3f} Å): Energy = {energy:.4f} eV")
        
        # Clean up calculator
        if hasattr(calc, 'destroy'):
            calc.destroy()
    
    all_lc_scan_data[name] = current_energies
    if not lc_values_template: # Capture LC values from the first material run
        lc_values_template = current_lcs

print("\n--- Lattice Constant Convergence calculations complete ---")

# --- Plotting Lattice Constant Convergence for all metals (single plot) ---
plt.figure(figsize=(12, 8))
for metal in materials:
    plt.plot(lc_values_template, all_lc_scan_data[metal], marker='o', linestyle='-', label=metal)

plt.xlabel('Lattice Constant (Å)', fontsize=14)
plt.ylabel('Total Energy (eV)', fontsize=14)
plt.title('GPAW PBE: Energy vs. Lattice Constant Convergence', fontsize=16)
plt.legend(fontsize=12, loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
print("Plot for PBE Lattice Constant Convergence displayed.")

print("\n--- All Convergence Tests Completed ---")