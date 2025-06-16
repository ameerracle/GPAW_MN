import matplotlib.pyplot as plt
import pandas as pd
from ase.io import read
from gpaw import GPAW, PW, FermiDirac
from dftd3.ase import DFTD3

# Materials and scaling setup
materials = ['ZrN', 'VN', 'NbN', 'TiN', 'ScN']
scales = [round(0.90 + 0.02 * i, 3) for i in range(5)] + [1.0] + [1.02]

# Initialize dataframe
data = {}

# GPAW calculator
def get_calc():
    return DFTD3(method="PBE", damping="d3op",
                calculator=GPAW(mode=PW(500),
                txt=None,
                xc='PBE',
                kpts={'size': (5, 5, 5)},
                occupations=FermiDirac(0.05)))
'''
def get_calc():
    return GPAW(mode=PW(500),
                xc='PBE',
                kpts={'size': (5, 5, 5)},
                occupations=FermiDirac(0.05),
                txt=None)
'''  

# Collect data
for name in materials:
    atoms = read(f"{name}.cif")  # or use .cif if needed
    original_cell = atoms.get_cell()
    
    lcs = []
    energies = []

    for scale in sorted(set(scales)):
        atoms.set_cell(original_cell * scale, scale_atoms=True)
        atoms.set_calculator(get_calc())
        energy = atoms.get_potential_energy()
        a = atoms.get_cell()[0, 0]
        lcs.append(a)
        energies.append(energy)
    
    data[name] = energies
    if 'LC' not in locals():
        LC = lcs

# Save to CSV
df = pd.DataFrame(data, index=LC)
df.index.name = 'Lattice Constant (Å)'
df.to_csv("lattice_energy_scan.csv")

# Plot
plt.figure()
for metal in materials:
    plt.plot(df.index, df[metal], label=metal)
plt.xlabel('Lattice Constant (Å)')
plt.ylabel('Total Energy (eV)')
plt.legend()
plt.grid(True)
plt.show()
