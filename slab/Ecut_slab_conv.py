import os
import numpy as np
import pandas as pd
from ase.io import read
from gpaw import GPAW, PW, FermiDirac
from ase.calculators.mixing import SumCalculator
from dftd3.ase import DFTD3
from dftd4.ase import DFTD4

# Ecut values to test (in eV)
ecut_values = [350, 400, 450, 500, 550, 600]
kpts = (5, 5, 1)
smearing = 0.05

# Scan current directory for all *_slab_100.xyz files
slab_files = [f for f in os.listdir() if f.endswith('_slab_100.xyz')]

def get_calculator(ecut, txt_file, functional):
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

# Loop over each slab and functional
for fname in slab_files:
    atoms = read(fname)
    name_base = fname.replace('_slab_100.xyz', '')
    parts = name_base.split('_')
    material = parts[0]
    functional = parts[1]

    energies = []

    print(f"\n?? Testing ECUT convergence for: {material} [{functional}]")

    for ecut in ecut_values:
        try:
            label = f"{material}_{functional}_ecut{ecut}"
            txt_file = f"{label}.txt"
            calc = get_calculator(ecut, txt_file, functional)
            atoms.calc = calc
            energy = atoms.get_potential_energy()
            energies.append((ecut, energy))
            print(f" Ecut = {ecut} eV ? Energy = {energy:.6f} eV")
        except Exception as e:
            print(f"  Ecut = {ecut} failed: {e}")
        finally:
            if hasattr(calc, 'clean'):
                calc.clean()

    # Save results to CSV
    df = pd.DataFrame(energies, columns=['Ecut (eV)', 'Energy (eV)'])
    csv_name = f"{material}_{functional}_ecut_convergence.csv"
    df.to_csv(csv_name, index=False)
    print(f"Saved results to {csv_name}")
