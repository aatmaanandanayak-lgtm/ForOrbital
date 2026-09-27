from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, AllChem

# Verified full Chl a structure from Phase 1
full_smiles = ("C=Cc1c(C)c2cc3nc(c4c5[n-]c(cc6nc(cc1[n-]2)C(C)=C6CC)c(C)c5C(=O)C4C(=O)OC)"
               "C(CCC(=O)OCC=C(C)CCCC(C)CCCC(C)CCCC(C)C)C3C.[Mg+2]")

# Truncate phytyl ester (OCC=C(C)CCCC(C)CCCC(C)CCCC(C)C) to methyl ester (OC)
truncated_smiles = ("C=Cc1c(C)c2cc3nc(c4c5[n-]c(cc6nc(cc1[n-]2)C(C)=C6CC)c(C)c5C(=O)C4C(=O)OC)"
                     "C(CCC(=O)OC)C3C.[Mg+2]")

full_mol = Chem.MolFromSmiles(full_smiles)
trunc_mol = Chem.MolFromSmiles(truncated_smiles)

print("Full Chl a:      ", rdMolDescriptors.CalcMolFormula(full_mol),
      " charge=", Chem.GetFormalCharge(full_mol),
      " heavy atoms=", full_mol.GetNumHeavyAtoms())
print("Truncated (Me-ester):", rdMolDescriptors.CalcMolFormula(trunc_mol),
      " charge=", Chem.GetFormalCharge(trunc_mol),
      " heavy atoms=", trunc_mol.GetNumHeavyAtoms())
print(f"\nHeavy atom reduction: {full_mol.GetNumHeavyAtoms()} -> {trunc_mol.GetNumHeavyAtoms()} "
      f"({100*(1-trunc_mol.GetNumHeavyAtoms()/full_mol.GetNumHeavyAtoms()):.0f}% smaller)")

# Sanity check: confirm the macrocycle + conjugated substituents are IDENTICAL
# between full and truncated - only the remote saturated tail changed
core_smarts = Chem.MolFromSmarts("c1ccc2cc3ccccc3cc2c1")  # rough aromatic core probe, just a sanity gate
