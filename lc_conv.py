from ase.io import read
from gpaw import GPAW, FermiDirac, PW
import numpy as np
import pandas as pd
from dftd3.ase import DFTD3
from dftd4.ase import DFTD4
from ase.units import kJ
from ase.eos import EquationOfState
import matplotlib.pyplot as plt

#metals = ['TiN', 'VN', 'ScN', 'ZrN', 'NbN']
metals = ['TiN']
ecut_value = 500
kpnts_mesh = (5, 5, 5)
csv_file = 'Lattice_constant_results.csv'
txt_file = 'Lattice_constant_results.txt'

# Initialize an empty list to store results for pandas DataFrame
results_data = []

# Open the text file for writing results
with open(txt_file, 'w') as f_txt:
    for metal in metals:
        filename = f'{metal}.cif'
        crystal = read(filename)
        print(f"Starting lattice constant calculation for {metal}...")
        f_txt.write(f"--- {metal} ---\n")

        # Define a range of lattice scaling factors
        # It's good practice to choose a range that covers potential variations
        scaling_factors = np.linspace(0.70, 0.9, 7) # 7 points from 0.95 to 1.05 of original lattice constant
        volumes = []
        energies = []

        try:
            for s_factor in scaling_factors:
                atoms = crystal.copy()
                # Scale the lattice vectors directly
                atoms.set_cell(atoms.get_cell() * s_factor, scale_atoms=True)

                # The GPAW calculator needs to be re-initialized for each metal
                # and potentially for each scaling factor if the grid changes significantly.
                # However, for lattice constant calculations, it's often more efficient
                # to define the calculator once with a flexible grid, or
                # to create a new calculator for each `atoms` object to ensure proper setup.
                # Here, we place the calculator initialization inside the loop for each atom object
                # to ensure the txt file name is unique for each metal and calculation.
                calc = DFTD4(method="PBE", damping="d3op", calculator=GPAW(mode=PW(ecut_value),
                                                                        xc='PBE',
                                                                        txt=f'{metal}_lc_D4_{s_factor:.2f}.txt', # Unique txt file for each scaling
                                                                        kpts={'size': kpnts_mesh},
                                                                        occupations=FermiDirac(0.05)))
                atoms.calc = calc
                energy = atoms.get_potential_energy()
                volume = atoms.get_volume()

                volumes.append(volume)
                energies.append(energy)
                f_txt.write(f"  Scaling Factor: {s_factor:.2f}, Volume: {volume:.4f} A^3, Energy: {energy:.4f} eV\n")
                print(f"  Scaling Factor: {s_factor:.2f}, Volume: {volume:.4f} A^3, Energy: {energy:.4f} eV")

            # Fit equation of state to find the equilibrium volume and bulk modulus
            eos = EquationOfState(volumes, energies)
            v0, e0, B = eos.fit()

            # Calculate equilibrium lattice constant from equilibrium volume
            # Assuming the initial crystal is cubic or near-cubic for simplicity
            # For non-cubic systems, this would be more complex and depend on the initial cell shape
            initial_volume = crystal.get_volume()
            initial_lattice_constant = (initial_volume)**(1/3.0)
            equilibrium_lattice_constant = initial_lattice_constant * (v0 / initial_volume)**(1/3.0)


            f_txt.write(f"  Equilibrium Volume (V0): {v0:.4f} A^3\n")
            f_txt.write(f"  Minimum Energy (E0): {e0:.4f} eV\n")
            f_txt.write(f"  Bulk Modulus (B): {B / kJ * 1.0e24:.4f} GPa\n") # Convert from eV/A^3 to GPa
            f_txt.write(f"  Equilibrium Lattice Constant: {equilibrium_lattice_constant:.4f} A\n\n")

            print(f"  Equilibrium Volume (V0): {v0:.4f} A^3")
            print(f"  Minimum Energy (E0): {e0:.4f} eV")
            print(f"  Bulk Modulus (B): {B / kJ * 1.0e24:.4f} GPa")
            print(f"  Equilibrium Lattice Constant: {equilibrium_lattice_constant:.4f} A")

            # Store results for DataFrame
            results_data.append({
                'Metal': metal,
                'Equilibrium Lattice Constant (A)': equilibrium_lattice_constant,
                'Equilibrium Volume (A^3)': v0,
                'Minimum Energy (eV)': e0,
                'Bulk Modulus (GPa)': B / kJ * 1.0e24
            })

            # Plotting the E-V curve
            plt.figure()
            eos.plot()
            plt.title(f'Energy vs Volume for {metal}')
            plt.xlabel('Volume ($\AA^3$)')
            plt.ylabel('Energy (eV)')
            plt.savefig(f'{metal}_eos.png')
            plt.close() # Close the plot to free memory

        except Exception as e:
            print(f"An error occurred for {metal}: {e}")
            f_txt.write(f"An error occurred for {metal}: {e}\n\n")

# Create a pandas DataFrame from the collected results
df_results = pd.DataFrame(results_data)

# Save the DataFrame to an Excel file
try:
    df_results.to_excel(csv_file, index=False)
    print(f"\nResults successfully written to {csv_file}")
except Exception as e:
    print(f"Error writing to Excel file: {e}")

print("All calculations finished.")