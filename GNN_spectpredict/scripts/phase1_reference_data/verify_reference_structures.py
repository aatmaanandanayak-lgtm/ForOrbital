"""
Verify the five chlorin-family reference structures (Phase 1).

Parses each compound's SMILES/InChI with RDKit and checks the computed
molecular formula and formal charge against literature values, BEFORE
trusting any structure downstream. Two real errors were caught this way
during Phase 1 - see notes below - which is why this check exists as a
standalone, re-runnable script rather than a one-off.
"""
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# (name, input_type, string, expected_formula, source, notes)
STRUCTURES = [
    ("Chlorophyll a", "smiles",
     "CCC1=C(C2=NC1=CC3=C(C4=C([N-]3)C(=C5C(C(C(=N5)C=C6C(=C(C(=C2)[N-]6)C=C)C)C)"
     "CCC(=O)OCC=C(C)CCCC(C)CCCC(C)CCCC(C)C)C(C4=O)C(=O)OC)C)C.[Mg+2]",
     "C55H72MgN4O5",
     "Wikidata Q133878 (mirrors PubChem CID 6266510)",
     "Clean on first pass."),
    ("Chlorophyll b", "smiles",
     "CCC1=C(C2=NC1=CC3=C(C4=C([N-]3)C(=C5C(C(C(=N5)C=C6C(=C(C(=C2)[N-]6)C=C)C)C)"
     "CCC(=O)OCC=C(C)CCCC(C)CCCC(C)CCCC(C)C)C(C4=O)C(=O)OC)C)C=O.[Mg+2]",
     "C55H70MgN4O6",
     "Wikidata Q1943387 (mirrors PubChem CID 11593175)",
     "Clean on first pass."),
    ("Chlorophyll d", "smiles",
     "CCC1=C(C2=NC1=CC3=C(C4=C([N-]3)C(=C5C(C(C(=N5)C=C6C(=C(C(=C2)[N-]6)C=O)C)C)"
     "CCC(=O)OCC=C(C)CCCC(C)CCCC(C)CCCC(C)C)C(C4=O)C(=O)OC)C)C.[Mg+2]",
     "C54H70MgN4O6",
     "Wikidata Q82182 (mirrors PubChem CID 16070025)",
     "Clean on first pass. Cross-checked independently against NCATS GSRS."),
    ("Chlorophyll f", "smiles",
     r"[H]C(=O)c1c(C=C)c2[n]3c1/C=C1/[C@@H](C)[C@H](CCC(=O)OC/C=C(\C)CCC[C@H](C)CCC"
     r"[C@H](C)CCCC(C)C)C4=[N]1->[Mg]31<-[N]3=C(/C=c5/c(C)c6c([n]51)=C4[C@@H](C(=O)OC)"
     r"C6=O)C(CC)=C(C)/C3=C/2",
     "C55H70MgN4O6",
     "ChEBI CHEBI:61290 (direct entry)",
     "IMPORTANT: the Wikidata-mirrored InChI for this compound (via PubChem "
     "CID 152743444) FAILS this check (RDKit mobile-H mismatch, comes back "
     "as +1 charge, not neutral). Use this ChEBI SMILES instead - it uses "
     "explicit dative-bond notation for the Mg coordination and parses "
     "cleanly."),
    ("Pheophytin a", "inchi",
     "InChI=1S/C55H74N4O5/c1-13-39-35(8)42-28-44-37(10)41(24-25-48(60)64-27-26-34(7)"
     "23-17-22-33(6)21-16-20-32(5)19-15-18-31(3)4)52(58-44)50-51(55(62)63-12)54(61)"
     "49-38(11)45(59-53(49)50)30-47-40(14-2)36(9)43(57-47)29-46(39)56-42/h13,26,28-33,"
     "37,41,51,56-57H,1,14-25,27H2,2-12H3",
     "C55H74N4O5",
     "ChEBI CHEBI:44898 (InChI field)",
     "IMPORTANT: the plain SMILES field on the same ChEBI page FAILS this "
     "check (2H short - missing both inner NH protons, since aromatic "
     "lowercase 'n' with no explicit [nH] was mis-parsed). This InChI "
     "carries the correct mobile-H layer; use it, not the raw SMILES field. "
     "NOTE: currently excluded from the active reference set - see project "
     "README, 'Reference set decision' - kept here for completeness and in "
     "case it's needed later."),
]


def verify(name, kind, s, expected_formula):
    mol = Chem.MolFromSmiles(s) if kind == "smiles" else Chem.MolFromInchi(s)
    if mol is None:
        return None, None, None
    formula = rdMolDescriptors.CalcMolFormula(mol)
    charge = Chem.GetFormalCharge(mol)
    formula_stripped = formula.split('+')[0].split('-')[0]
    ok = (formula_stripped == expected_formula) and (charge == 0)
    return formula, charge, ok


if __name__ == "__main__":
    print(f"{'Compound':<16}{'Formula':<18}{'Charge':<8}{'Expected':<16}{'Result'}")
    print("-" * 70)
    for name, kind, s, expected, source, notes in STRUCTURES:
        formula, charge, ok = verify(name, kind, s, expected)
        if formula is None:
            print(f"{name:<16}{'PARSE FAILED':<18}")
            continue
        status = "PASS" if ok else "CHECK"
        print(f"{name:<16}{formula:<18}{charge:<8}{expected:<16}{status}")
    print("\nSee source/notes fields in STRUCTURES for provenance and the")
    print("two errors this check caught (Chl f InChI, pheophytin a SMILES).")
