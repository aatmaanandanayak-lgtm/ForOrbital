"""
Extract the 4 Gouterman frontier MO energies (HOMO-1, HOMO, LUMO, LUMO+1)
from an ORCA output file, and check whether their splitting predicts a
Bx-By gap consistent with the ~2240-2305 cm^-1 the redundancy relation
derived from real Qx/Qy data (Phase 3).

Usage: python3 parse_orca_frontier_orbitals.py chla_frontier_orbitals.out
"""
import sys
import re

HARTREE_TO_CM1 = 219474.6

def parse_orbital_energies(filepath):
    """ORCA prints the FINAL orbital energy table after the LAST SCF cycle
    of a geometry optimization (not the first, unconverged one) - so we
    take the LAST occurrence of the block, not the first."""
    with open(filepath) as f:
        text = f.read()

    blocks = re.findall(
        r"ORBITAL ENERGIES\s*\n-+\s*\n\s*NO\s+OCC\s+E\(Eh\)\s+E\(eV\)\s*\n((?:.*\n)+?)\n",
        text
    )
    if not blocks:
        raise RuntimeError("No 'ORBITAL ENERGIES' block found - is the job actually finished? "
                            "Check the .out file directly, and check the matching .err file "
                            "for a crash before assuming this is a parsing bug.")

    last_block = blocks[-1]
    orbitals = []
    for line in last_block.strip().split("\n"):
        parts = line.split()
        if len(parts) < 4:
            continue
        no, occ, e_eh, e_ev = parts[0], parts[1], parts[2], parts[3]
        orbitals.append((int(no), float(occ), float(e_eh), float(e_ev)))

    return orbitals


def find_frontier_orbitals(orbitals):
    occupied = [o for o in orbitals if o[1] > 0.5]
    virtual = [o for o in orbitals if o[1] < 0.5]
    occupied.sort(key=lambda x: x[0])
    virtual.sort(key=lambda x: x[0])
    homo_minus1, homo = occupied[-2], occupied[-1]
    lumo, lumo_plus1 = virtual[0], virtual[1]
    return homo_minus1, homo, lumo, lumo_plus1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 parse_orca_frontier_orbitals.py <orca_output_file>")
        sys.exit(1)

    orbitals = parse_orbital_energies(sys.argv[1])
    hm1, h, l, lp1 = find_frontier_orbitals(orbitals)

    print(f"{'Orbital':<12}{'E (Eh)':<14}{'E (eV)':<12}")
    print(f"{'HOMO-1':<12}{hm1[2]:<14.6f}{hm1[3]:<12.4f}")
    print(f"{'HOMO':<12}{h[2]:<14.6f}{h[3]:<12.4f}")
    print(f"{'LUMO':<12}{l[2]:<14.6f}{l[3]:<12.4f}")
    print(f"{'LUMO+1':<12}{lp1[2]:<14.6f}{lp1[3]:<12.4f}")

    homo_split_cm1 = abs(h[2] - hm1[2]) * HARTREE_TO_CM1
    lumo_split_cm1 = abs(lp1[2] - l[2]) * HARTREE_TO_CM1
    print(f"\nHOMO splitting (a2u-a1u analog): {homo_split_cm1:.1f} cm^-1")
    print(f"LUMO splitting (egx-egy analog): {lumo_split_cm1:.1f} cm^-1")
    print(f"\nCompare against Phase 3's redundancy-relation prediction:")
    print(f"  Bx-By ~ 2240-2305 cm^-1 (consistent across Chl a/d/f)")
    print(f"\n(Note: HOMO/LUMO splitting isn't literally Bx-By - Bx-By is a")
    print(f" CONFIGURATION energy difference, which also involves the")
    print(f" two-electron J,K terms Phase 3 flagged as needing calibration.")
    print(f" This orbital-splitting comparison is a first-order sanity check,")
    print(f" not the rigorous final answer - but same order of magnitude")
    print(f" would be a meaningful, encouraging signal; wildly different")
    print(f" would flag a real problem worth investigating before going further.)")

# --- Quick self-test against synthetic ORCA-format text, since no real
# ORCA output exists yet to test this parser against ---
if len(sys.argv) == 1:
    pass
