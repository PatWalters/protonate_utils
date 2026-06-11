#!/usr/bin/env python

"""
Read a local PDB file, optionally remove a ligand by residue name,
add hydrogens with Hydride, and write the result to a PDB file.

Hydrogens are reordered so that each hydrogen immediately follows the
heavy atom to which it is bonded.

Usage
-----
    python add_hydrogens.py input.pdb AP5 output.pdb
    python add_hydrogens.py input.pdb none output.pdb   # no ligand removed
"""

import argparse

import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import hydride


def reorder_hydrogens_after_heavy_atoms(atoms):
    """
    Return a new AtomArray where each hydrogen immediately follows the
    heavy atom it is bonded to.

    Hydrogens bonded to the same heavy atom appear in the order Hydride
    originally placed them. Orphan hydrogens (no heavy-atom bond found)
    are appended at the end as a fallback.

    The input AtomArray must have a populated BondList.
    """
    if atoms.bonds is None:
        raise ValueError(
            "AtomArray must have an associated BondList; "
            "call connect_via_residue_names() before reordering."
        )

    # neighbors has shape (n_atoms, max_bonds); -1 entries are padding.
    neighbors, _ = atoms.bonds.get_all_bonds()
    is_hydrogen = atoms.element == "H"
    n_atoms = len(atoms)

    new_order = []
    placed = np.zeros(n_atoms, dtype=bool)

    for i in range(n_atoms):
        if placed[i] or is_hydrogen[i]:
            continue
        # Place the heavy atom.
        new_order.append(i)
        placed[i] = True
        # Then its bonded hydrogens, in their original relative order.
        h_indices = sorted(
            int(j) for j in neighbors[i]
            if j >= 0 and is_hydrogen[j] and not placed[j]
        )
        for j in h_indices:
            new_order.append(j)
            placed[j] = True

    # Any leftover atoms (e.g. unbonded hydrogens) go at the end.
    for i in range(n_atoms):
        if not placed[i]:
            new_order.append(i)
            placed[i] = True

    return atoms[new_order]


def prepare_structure(input_path, ligand_res_name, output_path,
                      ph=7.0, relax=True, quiet=False):
    # 1. Read the input PDB.
    structure = pdb.PDBFile.read(input_path).get_structure(model=1)

    # 2. Optionally remove the ligand by residue name (3-letter CCD code).
    #    "none" (any case) means "keep everything".
    if ligand_res_name is not None and ligand_res_name.lower() != "none":
        target = ligand_res_name.upper()
        keep_mask = np.char.upper(structure.res_name.astype(str)) != target
        if keep_mask.all():
            raise ValueError(
                f"No atoms with res_name '{target}' found in {input_path}."
            )
        structure = structure[keep_mask]

    # 3. Strip any pre-existing hydrogens; Hydride will add them itself.
    structure = structure[structure.element != "H"]

    # 4. Assign covalent bonds from CCD residue templates.
    structure.bonds = struc.connect_via_residue_names(structure)

    # 5. Set formal charges for canonical amino acids at the requested pH.
    charges = hydride.estimate_amino_acid_charges(structure, ph=ph)
    structure.set_annotation("charge", charges)

    # 6. Add hydrogens, then optionally relax their geometry.
    structure, _ = hydride.add_hydrogen(structure)
    if relax:
        structure.coord = hydride.relax_hydrogen(structure)

    # 7. Reorder so each hydrogen follows its bonded heavy atom.
    structure = reorder_hydrogens_after_heavy_atoms(structure)

    # 8. Write the result.
    out = pdb.PDBFile()
    out.set_structure(structure)
    out.write(output_path)
    if not quiet:
        print(f"Wrote hydrogenated structure to {output_path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_pdb", help="Path to the input PDB file")
    p.add_argument(
        "ligand_res_name",
        help="Residue name of the ligand to remove (e.g. ATP, HEM, AP5). "
             "Pass 'none' to skip ligand removal.",
    )
    p.add_argument("output_pdb", help="Path to the output PDB file")
    p.add_argument(
        "--ph", type=float, default=7.0,
        help="pH used to estimate amino-acid formal charges (default: 7.0)",
    )
    p.add_argument(
        "--no-relax", action="store_true",
        help="Skip dihedral relaxation of hydrogens.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_structure(
        input_path=args.input_pdb,
        ligand_res_name=args.ligand_res_name,
        output_path=args.output_pdb,
        ph=args.ph,
        relax=not args.no_relax,
    )
