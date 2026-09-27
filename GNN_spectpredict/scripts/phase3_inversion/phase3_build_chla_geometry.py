from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
import numpy as np

# Verified pheophytin a (metal-free) from Phase 1, truncated phytyl->methyl ester
pheo_full = ("C=Cc1c(C)c2cc3nc(c4c5nc(cc6[nH]c(cc1[nH]2)c(C)c6CC)C(C)=C5C(=O)C4C(=O)OC)"
             "C(CCC(=O)OCC=C(C)CCCC(C)CCCC(C)CCCC(C)C)C3C")
pheo_trunc = ("C=Cc1c(C)c2cc3nc(c4c5nc(cc6[nH]c(cc1[nH]2)c(C)c6CC)C(C)=C5C(=O)C4C(=O)OC)"
              "C(CCC(=O)OC)C3C")

mol = Chem.MolFromSmiles(pheo_trunc)
print("Truncated pheophytin (precursor):", rdMolDescriptors.CalcMolFormula(mol),
      " charge=", Chem.GetFormalCharge(mol), " heavy atoms=", mol.GetNumHeavyAtoms())

mol = Chem.AddHs(mol)
cid = AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
if cid < 0:
    raise RuntimeError("3D embedding failed")
res = AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
print(f"MMFF optimization converged: {res == 0}")

# Identify the 4 aromatic ring nitrogens and specifically the 2 that carry
# an N-H (pyrrole-type) - these H's get dropped when Mg takes their place
conf = mol.GetConformer()
ring_n_atoms = []
nh_hydrogens_to_remove = []
for atom in mol.GetAtoms():
    if atom.GetSymbol() == "N" and atom.GetIsAromatic():
        ring_n_atoms.append(atom.GetIdx())
        for nbr in atom.GetNeighbors():
            if nbr.GetSymbol() == "H":
                nh_hydrogens_to_remove.append(nbr.GetIdx())

print(f"\nFound {len(ring_n_atoms)} aromatic ring nitrogens (expect 4)")
print(f"Found {len(nh_hydrogens_to_remove)} N-H hydrogens to remove for Mg placement (expect 2)")

n_positions = np.array([list(conf.GetAtomPosition(i)) for i in ring_n_atoms])
mg_position = n_positions.mean(axis=0)
print(f"\nMg placed at centroid of 4 ring N atoms: {mg_position}")
print(f"N-Mg distances (should be roughly similar to each other, ~2.0-2.1 A is typical):")
for i, pos in zip(ring_n_atoms, n_positions):
    d = np.linalg.norm(pos - mg_position)
    print(f"  N(idx={i}): {d:.3f} A")

# Build final XYZ atom list: all atoms except the 2 N-H hydrogens, plus Mg
xyz_lines = []
for atom in mol.GetAtoms():
    if atom.GetIdx() in nh_hydrogens_to_remove:
        continue
    pos = conf.GetAtomPosition(atom.GetIdx())
    xyz_lines.append(f"{atom.GetSymbol():2s}  {pos.x:12.6f}  {pos.y:12.6f}  {pos.z:12.6f}")
xyz_lines.append(f"Mg  {mg_position[0]:12.6f}  {mg_position[1]:12.6f}  {mg_position[2]:12.6f}")

n_atoms_total = len(xyz_lines)
print(f"\nTotal atoms in final structure: {n_atoms_total}")

with open("/tmp/chla_truncated_geom.xyz", "w") as f:
    f.write(f"{n_atoms_total}\n")
    f.write("Chl a, phytyl truncated to methyl ester, Mg placed at ring-N centroid\n")
    f.write("\n".join(xyz_lines) + "\n")

print("Saved chla_truncated_geom.xyz")
