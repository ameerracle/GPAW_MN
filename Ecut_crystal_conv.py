from ase.io import read
from gpaw import GPAW, FermiDirac, PW
import numpy as np
import pandas as pd
from dftd4.ase import DFTD4

MN = ['TiN', 'VN', 'ScN', 'ZrN', 'NbN']

# Excel writer to save multiple sheets
output_excel = 'ecut_convergence_results.xlsx'
writer = pd.ExcelWriter(output_excel, engine='openpyxl')

for metal in MN:
    filename = f'{metal}.cif'  # Use current metal's CIF file
    ecut_values = np.linspace(300, 600, 8)  # 8 points from 300 to 600 eV

    kpts_mesh = (6, 6, 6)
    crystal = read(filename)  # Read crystal structure

    energies = []

    print(f"Starting Ecut convergence test for {metal}...")
    for ecut in ecut_values:
        print(f"  Calculating for Ecut = {ecut:.1f} eV...")
        # Set up GPAW calculator
        D4 = DFTD4(method="GPAW")
        calc = GPAW(mode=PW(ecut),
                    xc='PBE',
                    txt=f'{metal}_ecut_{ecut:.1f}.txt',
                    kpts={'size': kpts_mesh, 'gamma': True},
                    occupations=FermiDirac(0.05))

        crystal.calc = calc

        try:
            energy = crystal.get_potential_energy()
            energies.append(energy)
            print(f"    Total energy: {energy:.4f} eV")
        except Exception as e:
            print(f"    Error at Ecut = {ecut:.1f} eV: {e}")
            energies.append(np.nan)

        # Clean up
        del calc

    # Prepare dataframe and write to sheet
    df = pd.DataFrame({
        'Ecut (eV)': ecut_values,
        'Energy (eV)': energies
    })
    df.to_excel(writer, sheet_name=metal, index=False)

print("\nAll calculations finished. Saving results...")
writer.save()
writer.close()
print(f"Results saved to {output_excel}")
