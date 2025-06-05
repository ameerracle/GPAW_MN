from ase.io import read
from gpaw import GPAW, FermiDirac, PW
import numpy as np
import pandas as pd
from dftd4.ase import DFTD4
from dftd3.ase import DFTD3

MN = ['TiN', 'VN', 'ScN', 'ZrN', 'NbN']
ecut_values = np.linspace(300, 700, 8)
kpts_mesh = (6, 6, 6)

output_excel = 'ecut_convergence_results.xlsx'
output_txt = 'ecut_convergence_results.txt'

writer = pd.ExcelWriter(output_excel, engine='openpyxl')

functionals = {
    'BEEF': lambda ecut: GPAW(mode=PW(ecut),
                              xc='BEEF-vdW',
                              txt=None,
                              kpts={'size': kpts_mesh, 'gamma': True},
                              occupations=FermiDirac(0.05)),

    'PBE+D4': lambda ecut: DFTD4(method="GPAW", calculator=GPAW(mode=PW(ecut),
                                                                xc='PBE',
                                                                txt=None,
                                                                kpts={'size': kpts_mesh, 'gamma': True},
                                                                occupations=FermiDirac(0.05))),

    'PBE+D3': lambda ecut: DFTD3(method="PBE", damping="d3bj", calculator=GPAW(mode=PW(ecut),
                                                                               xc='PBE',
                                                                               txt=None,
                                                                               kpts={'size': kpts_mesh, 'gamma': True},
                                                                               occupations=FermiDirac(0.05)))
}

# Prepare to accumulate all results for backup txt file
all_results = []

for metal in MN:
    filename = f'{metal}.cif'
    crystal = read(filename)
    metal_results = []

    print(f"Starting Ecut convergence test for {metal}...")

    for functional_name, calculator_factory in functionals.items():
        for ecut in ecut_values:
            print(f"  {functional_name}: Ecut = {ecut:.1f} eV")
            try:
                atoms = crystal.copy()
                calc = calculator_factory(ecut)
                atoms.calc = calc
                energy = atoms.get_potential_energy()
                entry = {
                    'Metal': metal,
                    'Functional': functional_name,
                    'Ecut (eV)': ecut,
                    'Energy (eV)': energy
                }
                metal_results.append(entry)
                all_results.append(entry)
                print(f"    Total energy: {energy:.4f} eV")
                del calc
            except Exception as e:
                print(f"    Error for {functional_name} at Ecut = {ecut:.1f} eV: {e}")
                entry = {
                    'Metal': metal,
                    'Functional': functional_name,
                    'Ecut (eV)': ecut,
                    'Energy (eV)': np.nan
                }
                metal_results.append(entry)
                all_results.append(entry)

    df = pd.DataFrame(metal_results)
    df.to_excel(writer, sheet_name=metal, index=False)

# Save backup as .txt
txt_df = pd.DataFrame(all_results)
txt_df.to_csv(output_txt, sep='\t', index=False)

# Save Excel file
print("\nAll calculations finished. Saving results...")
writer.save()
writer.close()
print(f"Results saved to {output_excel} and backup to {output_txt}")
