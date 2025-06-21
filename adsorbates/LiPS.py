from ase.io import read, write
from ase.optimize import FIRE
from gpaw import GPAW, PW, FermiDirac
from dftd3.ase import DFTD3
from ase.build import molecule
import os

adsorbate_files = [
    'Li2S_Pople.xyz',
    'Li2S2_Pople.xyz',
    'Li2S4_Pople.xyz',
    'Li2S6_Pople.xyz',
    'Li2S8_Pople.xyz',
    'S8_Pople.xyz'
]

ecut = 500
kpts = (1, 1, 1)
fmax = 0.05
vacuum = 8.0  # angstrom of vacuum in each direction

output_dir = "optimized_adsorbates"
os.makedirs(output_dir, exist_ok=True)

for filename in adsorbate_files:
    atoms = atoms = read(filename)
    name = os.path.splitext(filename)[0]

    # Center molecule in cell with vacuum and enforce PBC
    atoms.center(vacuum=vacuum)
    atoms.set_pbc([True, True, True])  # Enforce PBC in all directions

    for functional in ['BEEF', 'PBE+D3']:
        tag = f"{name}_{functional.replace('+', '')}"
        txt_log = os.path.join(output_dir, f"{tag}.txt")
        traj_file = os.path.join(output_dir, f"{tag}.traj")
        xyz_file = os.path.join(output_dir, f"{tag}_opt.xyz")

        if functional == 'BEEF':
            calc = GPAW(mode=PW(ecut),
                        xc='BEEF-vdW',
                        kpts={'size': kpts, 'gamma': True},
                        occupations=FermiDirac(0.1),
                        txt=txt_log)

        elif functional == 'PBE+D3':
            base_calc = GPAW(mode=PW(ecut),
                             xc='PBE',
                             kpts={'size': kpts, 'gamma': True},
                             occupations=FermiDirac(0.1),
                             txt=txt_log)
            calc = DFTD3(base_calc, method='D3')

        atoms.calc = calc
        dyn = FIRE(atoms, trajectory=traj_file)
        dyn.run(fmax=fmax)
        write(xyz_file, atoms)
