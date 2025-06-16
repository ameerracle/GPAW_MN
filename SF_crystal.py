from ase.io import read, write
from gpaw import GPAW, FermiDirac, PW
import numpy as np
import pandas as pd
from dftd4.ase import DFTD4
from dftd3.ase import DFTD3
from ase.constraints import StrainFilter
from ase.optimize import FIRE
from ase.calculators.mixing import SumCalculator
import os

# --- Global Parameters ---
MN = ['TiN']
#MN = ['TiN', 'VN', 'ScN', 'ZrN', 'NbN']
ecut = 500  # eV, static value
kpts_mesh = (5, 5, 5)
smearing_width = 0.05 # Fermi-Dirac smearing width

output_csv = 'optimization_results.csv' # <--- Changed to .csv
output_txt = 'optimization_summary.txt'

# --- Function to create a GPAW calculator ---
def get_gpaw_calculator(txt_file_name, xc_functional='PBE'):
    """
    Returns a GPAW calculator with specified parameters.
    The SCF output will go to txt_file_name.
    """
    calc = GPAW(mode=PW(ecut),
                xc=xc_functional,
                txt=txt_file_name,
                kpts={'size': kpts_mesh},
                occupations=FermiDirac(smearing_width),
                verbose=2
                )
    return calc

# --- No need for pd.ExcelWriter anymore ---
# The CSV file will be written directly at the end.
if os.path.exists(output_csv): # <--- Check for .csv file
    os.remove(output_csv)
# No need to initialize writer = pd.ExcelWriter(...)

# List to store all results for DataFrame creation
all_optimization_data = []

# --- Main Loop for Metals and Functionals ---
for metal in MN:
    filename = f'{metal}.cif'
    if not os.path.exists(filename):
        print(f"Error: CIF file '{filename}' not found. Skipping {metal}.")
        continue

    initial_crystal = read(filename)
    print(f"\n--- Processing metal: {metal} ---")

    # --- Define Calculators for the current metal ---
    # PBE Calculator
    pbe_gpaw_txt = f'{metal}_PBE_gpaw.txt'
    pbe_calc = get_gpaw_calculator(pbe_gpaw_txt, xc_functional='PBE')

    # PBE+DFTD3 Calculator
    d3_gpaw_txt = f'{metal}_PBE_DFTD3_gpaw.txt'
    gpaw_part_for_d3 = get_gpaw_calculator(d3_gpaw_txt, xc_functional='PBE')
    dftd3_part = DFTD3(method="PBE", damping="d3bj")
    pbe_d3_calc = SumCalculator([gpaw_part_for_d3, dftd3_part])

    # PBE+DFTD4 Calculator
    d4_gpaw_txt = f'{metal}_PBE_DFTD4_gpaw.txt'
    gpaw_part_for_d4 = get_gpaw_calculator(d4_gpaw_txt, xc_functional='PBE')
    dftd4_part = DFTD4(method="PBE")
    pbe_d4_calc = SumCalculator([gpaw_part_for_d4, dftd4_part])

    # BEEF-vdW Calculator (standalone, no SumCalculator with DFT-D)
    beef_gpaw_txt = f'{metal}_BEEFvdW_gpaw.txt'
    beef_calc = get_gpaw_calculator(beef_gpaw_txt, xc_functional='BEEF-vdW')

    # Dictionary of functionals to iterate
    functional_setups = [
        ("PBE", pbe_calc),
        ("PBE+DFTD3", pbe_d3_calc),
        ("PBE+DFTD4", pbe_d4_calc),
        ("BEEF-vdW", beef_calc)
    ]

    for func_name, calculator_instance in functional_setups:
        print(f"  Functional: {func_name}")

        crystal_to_optimize = initial_crystal.copy()
        crystal_to_optimize.calc = calculator_instance

        traj_filename = f'{metal}_{func_name.replace("+", "_").replace("-", "_")}_opt.traj'
        log_filename = f'{metal}_{func_name.replace("+", "_").replace("-", "_")}_opt.log'
        print(f"    Optimizer Trajectory: {traj_filename}, Log: {log_filename}")

        sf = StrainFilter(crystal_to_optimize)
        opt = FIRE(sf, trajectory=traj_filename, logfile=log_filename)

        print(f"    Starting optimization for {metal} with {func_name}...")
        try:
            opt.run(fmax=0.02)
            print(f"    Optimization for {metal} with {func_name} completed.")
            
            # --- Collect Results ---
            final_energy = crystal_to_optimize.get_potential_energy()
            final_cell = crystal_to_optimize.get_cell()
            final_volume = crystal_to_optimize.get_volume()
            
            gpaw_output_file = ""
            if func_name == "PBE":
                gpaw_output_file = pbe_gpaw_txt
            elif func_name == "PBE+DFTD3":
                gpaw_output_file = d3_gpaw_txt
            elif func_name == "PBE+DFTD4":
                gpaw_output_file = d4_gpaw_txt
            elif func_name == "BEEF-vdW":
                gpaw_output_file = beef_gpaw_txt

            results_row = {
                'Metal': metal,
                'Functional': func_name,
                'Converged': True,
                'Final Energy (eV)': final_energy,
                'Final Volume (A^3)': final_volume,
                'Final Cell (Angstroms)': str(final_cell.array.tolist()),
                'Optimization Log': log_filename,
                'GPAW txt file': gpaw_output_file
            }
            all_optimization_data.append(results_row)

        except Exception as e:
            print(f"    Error during optimization for {metal} with {func_name}: {e}")
            gpaw_output_file = ""
            if func_name == "PBE":
                gpaw_output_file = pbe_gpaw_txt
            elif func_name == "PBE+DFTD3":
                gpaw_output_file = d3_gpaw_txt
            elif func_name == "PBE+DFTD4":
                gpaw_output_file = d4_gpaw_txt
            elif func_name == "BEEF-vdW":
                gpaw_output_file = beef_gpaw_txt

            results_row = {
                'Metal': metal,
                'Functional': func_name,
                'Converged': False,
                'Final Energy (eV)': np.nan,
                'Final Volume (A^3)': np.nan,
                'Final Cell (Angstroms)': 'Error',
                'Optimization Log': log_filename,
                'GPAW txt file': gpaw_output_file
            }
            all_optimization_data.append(results_row)
        finally:
            try:
                if hasattr(calculator_instance, 'destroy'):
                    calculator_instance.destroy()
                elif isinstance(calculator_instance, SumCalculator):
                    # Use 'calcs' instead of 'calculators'
                    for sub_calc in calculator_instance.calcs:
                        if hasattr(sub_calc, 'destroy'):
                            sub_calc.destroy()
            except Exception as cleanup_error:
                print(f"    Warning: Error during calculator cleanup: {cleanup_error}")
            
            try:
                del crystal_to_optimize, sf, opt
            except Exception as del_error:
                print(f"    Warning: Error during object deletion: {del_error}")
            
            print(f"    Cleanup for {metal} with {func_name} done.")

# --- Write results to CSV and Text file ---
results_df = pd.DataFrame(all_optimization_data)

# Write to CSV
results_df.to_csv(output_csv, index=False) # <--- Changed to to_csv()
print(f"\nResults written to {output_csv}")

# Write to a simple text file
with open(output_txt, 'w') as f:
    f.write("Structural Optimization Summary\n")
    f.write("=" * 40 + "\n\n")
    f.write(results_df.to_string(index=False))
print(f"Summary written to {output_txt}")

print("\n--- All optimizations completed ---")
print("Check the generated .txt and .traj files for detailed optimization steps and GPAW output.")