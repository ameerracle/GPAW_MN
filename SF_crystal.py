from ase.io import read
from gpaw import GPAW, FermiDirac, PW
import numpy as np
import pandas as pd
from dftd4.ase import DFTD4 # Note: DFTD4 is imported but not used in your provided functionals
from dftd3.ase import DFTD3
from ase.constraints import StrainFilter
from ase.optimize import FIRE2


MN = ['TiN', 'VN', 'ScN', 'ZrN', 'NbN']
# ecut is now a static value and does not need to be passed as an argument
ecut = 500
kpts_mesh = (5, 5, 5)

output_excel = 'ecut_convergence_results.xlsx'
output_txt = 'ecut_convergence_results.txt'

# ExcelWriter is initialized but not used for writing results in the current loop.
# If you plan to save data, you'll need to add writing logic within the loop
# and remember to close the writer.
writer = pd.ExcelWriter(output_excel, engine='openpyxl')

# Functional definitions are simplified; ecut is directly used without lambda
functionals = {
    'BEEF': GPAW(mode=PW(ecut),
                 xc='BEEF-vdW',
                 txt=f'BEEF.txt',
                 kpts={'size': kpts_mesh, 'gamma': True},
                 occupations=FermiDirac(0.05)),

    'PBE+D3': DFTD3(method="PBE", damping="d3bj", calculator=GPAW(mode=PW(ecut),
                                                                 xc='PBE',
                                                                 txt=f'PBE_D3.txt', 
                                                                 kpts={'size': kpts_mesh, 'gamma': True},
                                                                 occupations=FermiDirac(0.05)))
}

for metal in MN:
    filename = f'{metal}.cif'
    initial_crystal = read(filename)  # Read the base structure once per metal
    print(f"Processing metal: {metal}")

    # Now, calculator_instance is directly the GPAW or DFTD3 object
    for func_name, calculator_instance in functionals.items():
        print(f"  Functional: {func_name}")

        crystal_to_optimize = initial_crystal.copy()  # Work on a copy for each functional

        # Set the pre-instantiated calculator
        crystal_to_optimize.calc = calculator_instance

        # Define unique filenames for trajectory and log files
        traj_filename = f'{metal}_{func_name}_strain.traj'
        log_filename = f'{metal}_{func_name}_strain.log'
        print(f"    Trajectory: {traj_filename}, Log: {log_filename}")

        # Set up the optimizer
        sf = StrainFilter(crystal_to_optimize)
        opt = FIRE2(sf, trajectory=traj_filename, logfile=log_filename)

        print(f"    Starting optimization...")
        try:
            opt.run(fmax=0.02)  # Force convergence criterion in eV/Å
            print(f"    Optimization for {metal} with {func_name} completed.")
            # Optionally, save the final optimized structure:
            # from ase.io import write
            # write(f'{metal}_{func_name}_optimized.cif', crystal_to_optimize)
        except Exception as e:
            print(f"    Error during optimization for {metal} with {func_name}: {e}")
        finally:

            if hasattr(calculator_instance, 'destroy'): # GPAW specific cleanup
                calculator_instance.destroy()
            del crystal_to_optimize, sf, opt
    print("-" * 30) # Separator for metals

# Remember to close the Excel writer if you add data writing logic.
# writer.close()