import os
import numpy as np
import pandas as pd
from ase.io import read
from gpaw import GPAW, PW, FermiDirac
from ase.calculators.mixing import SumCalculator
from dftd3.ase import DFTD3
from dftd4.ase import DFTD4

ecut = 500
smearing = 0.05
kpoint_list = [(4, 4, 1), (5, 5, 1), (6, 6, 1), (7, 7, 1), (8, 8, 1), (9, 9, 1)]
slab_files = [f for f in os.listdir() if f.endswith('_slab_100.xyz')]

def get_calculator(kpts, txt_file, functional):
    if functional == 'D3':
        return SumCalculator([
            GPAW(mode=PW(ecut), xc='PBE', kpts=kpts,
                 occupations=FermiDirac(smearing), txt=txt_file),
            DFTD3(method="PBE", damping="d3bj")
        ])
    elif functional == 'D4':
        return SumCalculator([
            GPAW(mode=PW(ecut), xc='PBE', kpts=kpts,
                 occupations=FermiDirac(smearing), txt=txt_file),
            DFTD4(method="PBE")
        ])
    elif functional == 'BEEF':
        return GPAW(mode=PW(ecut), xc='BEEF-vdW', kpts=kpts,
                    occupations=FermiDirac(smearing), txt=txt_file)
    else:
        raise ValueError(f"Unsupported functional: {functional}")

for fname in slab_files:
    atoms = read(fname)
    atoms.set_pbc([True, True, True])
    name_base = fname.replace('_slab_100.xyz', '')
    parts = name_base.split('_')
    material = parts[0]
    functional = parts[1]

    energies = []

    print(f"\nTesting K-point convergence for: {material} [{functional}]")

    for kpts in kpoint_list:
        try:
            klabel = f"{kpts[0]}x{kpts[1]}x{kpts[2]}"
            txt_file = f"{material}_{functional}_kp_{klabel}.txt"
            calc = get_calculator(kpts, txt_file, functional)
            atoms.calc = calc
            energy = atoms.get_potential_energy()
            energies.append((klabel, energy))
            print(f"  K-points = {klabel} ? Energy = {energy:.6f} eV")
        except Exception as e:
            print(f"  K-points {kpts} failed: {e}")
        finally:
            if hasattr(calc, 'clean'):
                calc.clean()

    df = pd.DataFrame(energies, columns=['Kpoints', 'Energy (eV)'])
    csv_name = f"{material}_{functional}_kpoints_convergence.csv"
    df.to_csv(csv_name, index=False)
    print(f"Saved results to {csv_name}")
