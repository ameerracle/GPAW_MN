#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os
from ase.io import read, write
from ase.io.trajectory import Trajectory
from gpaw import GPAW, FermiDirac, PW
from ase import Atoms
from dftd3.ase import DFTD3
from dftd4.ase import DFTD4
from ase.calculators.mixing import SumCalculator

# Best LCs from previous EOS fit
best_lcs = {
    'TiN': {'PBE+D3': 4.224825, 'PBE+D4': 4.222713, 'BEEF-vdW': 4.265981},
    'VN': {'PBE+D3': 4.105173, 'PBE+D4': 4.099841, 'BEEF-vdW': 4.154965},
    'ScN': {'PBE+D3': 4.475153, 'PBE+D4': 4.465668, 'BEEF-vdW': 4.516714},
    'ZrN': {'PBE+D3': 4.560577, 'PBE+D4': 4.561882, 'BEEF-vdW': 4.599342},
    'NbN': {'PBE+D3': 4.383374, 'PBE+D4': 4.381987, 'BEEF-vdW': 4.425092}
}

materials = list(best_lcs.keys())
functionals = ['PBE+D3', 'PBE+D4', 'BEEF-vdW']
ecut = 550
kpts_mesh = (11, 11, 11)
smearing_width = 0.05

# --- Calculator Setup ---
def get_gpaw_calculator(txt_file, xc_functional='PBE'):
    return GPAW(
        mode=PW(ecut), xc=xc_functional,
        kpts={'size': kpts_mesh},
        occupations={'name': 'fermi-dirac', 'width': smearing_width},
        txt=txt_file
    )

def setup_calculators(name):
    return {
        'PBE+D3': SumCalculator([
            get_gpaw_calculator(f'{name}_PBE_DFTD3_gpaw.txt'),
            DFTD3(method="PBE", damping="d3bj")
        ]),
        'PBE+D4': SumCalculator([
            get_gpaw_calculator(f'{name}_PBE_DFTD4_gpaw.txt'),
            DFTD4(method="PBE")
        ]),
        'BEEF-vdW': get_gpaw_calculator(f'{name}_BEEF_vdW_gpaw.txt', 'BEEF-vdW')
    }

# --- Calculation and Tracking ---
def calculate_and_record(atoms, calc, func, scale, mat, traj_writer, backup_holder):
    atoms = atoms.copy()
    atoms.calc = calc
    e = atoms.get_potential_energy()
    lc = atoms.cell.cellpar()[0]
    if e < backup_holder['energy']:
        backup_holder.update({'energy': e, 'atoms': atoms.copy(), 'scale': scale, 'lc': lc})
    traj_writer.write(atoms, energy=e)
    return {'Material': mat, 'Functional': func, 'Scale_Factor': scale, 'Lattice_Constant': lc, 'Energy': e, 'Energy_per_atom': e/len(atoms)}

# --- Main Execution ---
def main():
    results = {f: [] for f in functionals}

    for mat in materials:
        base = read(f'{mat}.cif')
        a0 = base.cell.cellpar()[0]
        calcs = setup_calculators(mat)

        for func in functionals:
            lc_best = best_lcs[mat][func]
            lc_range = np.linspace(lc_best - 0.2, lc_best + 0.2, 7)

            traj_file = f'{mat}_{func.lower().replace("+","_").replace("-","_")}_lc_scan.traj'
            backup = {'energy': 1e99, 'atoms': None, 'scale': None, 'lc': None}

            with Trajectory(traj_file, mode='w') as traj:
                for lc in lc_range:
                    scale = lc / a0
                    scaled = base.copy()
                    scaled.set_cell(base.get_cell() * scale, scale_atoms=True)
                    res = calculate_and_record(scaled, calcs[func], func, scale, mat, traj, backup)
                    results[func].append(res)

            if backup['atoms'] is not None:
                xyz_file = f'{mat}_{func}_best.xyz'
                write(xyz_file, backup['atoms'])
                print(f"✔ Saved best structure: {xyz_file} at LC = {backup['lc']:.4f} Å")

    # Save results
    for func in functionals:
        df = pd.DataFrame(results[func])
        if not df.empty:
            fname = func.lower().replace('+','_').replace('-','_')
            df.to_csv(f'{fname}_results.csv', index=False)
            print(f"✔ Saved results CSV: {fname}_results.csv")

if __name__ == '__main__':
    main()
