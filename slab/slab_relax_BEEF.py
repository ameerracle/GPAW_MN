from ase.io import read, write
from ase.constraints import FixAtoms
from ase.optimize import FIRE2
from gpaw import GPAW, PW, FermiDirac
import os

ecut = 500
kpts = (4, 4, 1)
smearing = 0.05
fmax = 0.02  # force threshold

# All slabs expected in format: 'TiN_D3_slab_100.xyz'
slab_files = [f for f in os.listdir() if '_BEEF_slab_100.xyz' in f]

def get_calculator(name, functional):
    txt_file = f'{name}_{functional}_slab_relax.txt'
    if functional == 'BEEF':
        return GPAW(
            mode=PW(ecut), xc='BEEF-vdW',  # Note: correct spelling is 'BEEF-vdW'
            kpts=kpts,
            occupations=FermiDirac(smearing),
            txt=txt_file
        )
    else:
        raise ValueError(f"Unsupported functional in file: {functional}")

for fname in slab_files:
    atoms = read(fname)
    basename = fname.replace('_slab_100.xyz', '')  # e.g., TiN_D3
    name_parts = basename.split('_')
    material = name_parts[0]
    functional = name_parts[1]  # D3, D4, or BEEF

    # Fix bottom 2 layers by z-coordinate
    z_positions = [atom.position[2] for atom in atoms]
    sorted_indices = sorted(range(len(atoms)), key=lambda i: z_positions[i])
    num_fixed = len(atoms) // 2  # fix bottom half (usually 2/4 layers)
    fix_indices = sorted_indices[:num_fixed]
    atoms.set_constraint(FixAtoms(indices=fix_indices))

    # Set calculator
    atoms.calc = get_calculator(material, functional)

    # Run relaxation
    traj_file = fname.replace('.xyz', '_relax.traj')
    dyn = FIRE2(atoms, trajectory=traj_file)
    dyn.run(fmax=fmax)

 
