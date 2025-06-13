from ase.io import read
from gpaw import GPAW, FermiDirac, PW
import numpy as np
import pandas as pd
from dftd3.ase import DFTD3

metals = ['TiN', 'VN', 'ScN', 'ZrN', 'NbN']
ecut_value = 500
kpnts_mesh = (5, 5, 5)
csv_file = 'Lattice_constant_results.csv'
txt_file = 'Lattice_constant_results.txt'
writer = pd.ExcelWriter(csv_file, engine='openpyxl')

calc = DFTD3(method="PBE", damping="d3bj", calculator=GPAW(mode=PW(ecut_value),
                                                    xc='PBE',
                                                    txt=f'{metal}_lc.txt',
                                                    kpts={'size': kpnts_mesh},
                                                    occupations=FermiDirac(0.05)))
for metal in metals:
    filename = f'{metal}.cif'
    crystal = read(filename)
    print(f"Starting lattice constant calculation for {metal}...")
    try:
        atoms = crystal.copy()
        atoms.calc = calc

        
        
