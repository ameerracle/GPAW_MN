from ase.io import read, write
from ase.constraints import FixAtoms
from ase.optimize import FIRE2
from gpaw import GPAW, PW, FermiDirac
from ase.calculators.mixing import SumCalculator
from dftd3.ase import DFTD3
import os

ecut = 500
kpts = (3, 3, 1)
smearing = 0.05
fmax = 0.02  # force threshold

# Define materials and functionals
materials = ['ScN', 'NbN']
functionals = ['BEEF', 'D3']

# Generate slab files list
slab_files = []
for material in materials:
    for functional in functionals:
        # Expected file format: Material_Li2S_ads_Functional.xyz
        filename = f'{material}_Li2S_ads_{functional}.xyz'
        if os.path.exists(filename):
            slab_files.append(filename)
        else:
            print(f"Warning: {filename} not found, skipping...")

def get_calculator(material, functional):
    """Get the appropriate calculator based on material and functional"""
    txt_file = f'{material}_{functional}_Li2S_ads_relax.txt'
    
    if functional == 'BEEF':
        return GPAW(
            mode=PW(ecut), 
            xc='BEEF-vdW',
            kpts=kpts,
            occupations=FermiDirac(smearing),
            txt=txt_file
        )
    elif functional == 'D3':
        return SumCalculator([
            GPAW(mode=PW(ecut), xc='PBE', kpts=kpts,
                 occupations=FermiDirac(smearing), txt=txt_file),
            DFTD3(method="PBE", damping="d3bj")
        ])
    else:
        raise ValueError(f"Unsupported functional: {functional}")

def fix_bottom_layers(atoms):
    """Fix the bottom half of the slab (typically bottom 2 layers)"""
    z_positions = [atom.position[2] for atom in atoms]
    sorted_indices = sorted(range(len(atoms)), key=lambda i: z_positions[i])
    num_fixed = len(atoms) // 2  # fix bottom half
    fix_indices = sorted_indices[:num_fixed]
    atoms.set_constraint(FixAtoms(indices=fix_indices))
    return atoms

# Main calculation loop
print("Starting Li2S adsorbate relaxation calculations...")
print(f"Found {len(slab_files)} slab files to process")

for fname in slab_files:
    print(f"\nProcessing: {fname}")
    
    # Parse filename to extract material and functional
    # Expected format: 'Material_Li2S_ads_Functional.xyz'
    base_name = os.path.basename(fname)
    name_parts = base_name.replace('.xyz', '').split('_')
    
    if len(name_parts) >= 4:
        material = name_parts[0]  # ScN, NbN, etc.
        functional = name_parts[3]  # BEEF or D3
    else:
        print(f"Error: Cannot parse filename {fname}, skipping...")
        continue
    
    try:
        # Read the slab with Li2S adsorbate
        atoms = read(fname)
        print(f"  Material: {material}, Functional: {functional}")
        print(f"  Number of atoms: {len(atoms)}")
        
        # Fix bottom layers
        atoms = fix_bottom_layers(atoms)
        print(f"  Fixed bottom {len(atoms.constraints[0].index)} atoms")
        
        # Set calculator
        atoms.calc = get_calculator(material, functional)
        
        # Set up output files
        traj_file = fname.replace('.xyz', '_relaxed.traj')
        final_file = fname.replace('.xyz', '_relaxed_final.xyz')
        
        print(f"  Output trajectory: {traj_file}")
        print(f"  Starting relaxation...")
        
        # Run relaxation
        dyn = FIRE2(atoms, trajectory=traj_file)
        dyn.run(fmax=fmax)
        
        # Save final structure
        #write(final_file, atoms)
        
        print(f"  ✓ Relaxation completed for {material}_{functional}")
        print(f"  Final energy: {atoms.get_potential_energy():.4f} eV")
        
    except Exception as e:
        print(f"  ✗ Error processing {fname}: {e}")
        continue

print("\nAll calculations completed!")

# Summary of expected output files:
print("\nExpected output files:")
for material in materials:
    for functional in functionals:
        base = f'{material}_{functional}_slab_100_relax_Li2S'
        print(f"  {base}_relaxed.traj (trajectory)")
        print(f"  {base}_relaxed_final.xyz (final structure)")
        print(f"  {material}_{functional}_Li2S_ads_relax.txt (GPAW output)")