#!/usr/bin/env python
"""
Protonation utilities for both ligands and proteins, selected by the
first command-line argument:

    protonate_utils.py ligand  ...    # small molecules (SDF/SMILES)
    protonate_utils.py protein ...    # protein structures (PDB)

Ligand mode
-----------
Add hydrogens to small molecules at a target pH (default 7.4). Input and
output may be either SDF or SMILES; the format is inferred from the file
extension (.smi/.smiles for SMILES, otherwise SDF).

pH-aware protonation states are predicted with Dimorphite-DL, the
resulting formal charges *and* H counts are mapped back onto the
original molecule via substructure matching, and explicit hydrogens are
added with Chem.AddHs(addCoords=True). When the input has 3D coordinates
(SDF), heavy-atom positions are not disturbed; only the newly added
hydrogens get computed positions based on the existing geometry. SMILES
input has no coordinates, so protonation is applied without any geometry.

Protein mode
------------
Read a local PDB file, optionally remove a ligand by residue name, add
hydrogens with Hydride at a target pH (default 7.0), and write the
result to a PDB file. Hydrogens are reordered so that each hydrogen
immediately follows the heavy atom to which it is bonded.

Install:
    pip install rdkit dimorphite-dl          # ligand mode
    pip install biotite hydride numpy        # protein mode

Usage:
    protonate_utils.py ligand input.sdf output.sdf
    protonate_utils.py ligand input.smi output.smi --ph 7.4
    protonate_utils.py protein input.pdb AP5 output.pdb
    protonate_utils.py protein input.pdb none output.pdb --ph 7.0
"""

import argparse
import sys


# ---------------------------------------------------------------------------
# Ligand mode (RDKit + Dimorphite-DL)
# ---------------------------------------------------------------------------

def _neutralized_copy(mol):
    """
    Return a copy of `mol` with all formal charges and explicit H counts
    zeroed, for use as a charge-insensitive substructure-match template.
    Sanitization is best-effort; if neutralization produces an invalid
    valence, we fall back to a partial sanitize that still updates ring
    info (which substructure matching uses).
    """
    from rdkit import Chem

    m = Chem.RWMol(mol)
    for a in m.GetAtoms():
        a.SetFormalCharge(0)
        a.SetNumExplicitHs(0)
        a.SetNoImplicit(False)
    try:
        Chem.SanitizeMol(m)
    except Exception:
        Chem.SanitizeMol(
            m,
            sanitizeOps=(
                Chem.SanitizeFlags.SANITIZE_ALL
                ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
            ),
        )
    return m


def _pick_state(input_smiles, states):
    """
    Choose one microstate from Dimorphite-DL's output deterministically.

    Dimorphite returns every microstate whose pKa falls within the
    requested pH window. Its list order is not stable across Python
    runs, so taking `states[0]` makes the pipeline non-deterministic:
    e.g. a secondary alkyl amide can come back as either NH or N-, and
    we'd silently flip between them on re-runs.

    Prefer the state whose total formal charge is closest to the input
    molecule's charge (with SMILES string as a deterministic tiebreak).
    For groups with pKa far from the pH window dimorphite returns a
    single state, so this only matters when dimorphite is unsure -- in
    which case "match the input charge" is a sensible default.
    """
    from rdkit import Chem

    target = sum(
        a.GetFormalCharge() for a in Chem.MolFromSmiles(input_smiles).GetAtoms()
    )

    def score(smi):
        m = Chem.MolFromSmiles(smi)
        charge = sum(a.GetFormalCharge() for a in m.GetAtoms()) if m else 10**6
        return (abs(charge - target), smi)

    return min(states, key=score)


def _target_atom_states(mol_heavy, ph):
    """
    Use Dimorphite-DL to determine the dominant protonation state at
    `ph`, then return a dict {atom_idx: (formal_charge, total_num_hs)}
    aligned to the atom indices of `mol_heavy`.

    Returning the total H count along with the charge is important: for
    aromatic heterocycles, charge alone underspecifies the atom and
    RDKit will fail to kekulize after the change. The template's H count
    fully constrains the bonding state.
    """
    from rdkit import Chem
    from dimorphite_dl import protonate_smiles

    smiles = Chem.MolToSmiles(mol_heavy)
    states = protonate_smiles(smiles, ph_min=ph - 0.5, ph_max=ph + 0.5)
    if not states:
        raise RuntimeError(f"Dimorphite-DL returned no states for {smiles!r}")

    chosen = _pick_state(smiles, states)
    template = Chem.MolFromSmiles(chosen)
    if template is None:
        raise RuntimeError(
            f"RDKit could not parse Dimorphite-DL output {chosen!r}"
        )

    # Map heavy-atom indices between original and template via a
    # charge-stripped substructure match (so e.g. -COOH still matches -COO-).
    match = _neutralized_copy(mol_heavy).GetSubstructMatch(
        _neutralized_copy(template)
    )
    if not match:
        raise RuntimeError(
            "Could not align protonation template with input molecule"
        )

    out = {}
    for template_idx, orig_idx in enumerate(match):
        ta = template.GetAtomWithIdx(template_idx)
        out[orig_idx] = (ta.GetFormalCharge(), ta.GetTotalNumHs())
    return out


def protonate_molecule(mol, ph, add_coord_hs=True):
    """
    Return a Mol with pH-appropriate protonation.

    When the input carries a 3D conformer and `add_coord_hs` is set,
    explicit hydrogens are added and positioned from the existing
    geometry while the heavy-atom coordinates are preserved (this is the
    SDF-output path). Otherwise protonation is left implicit, which is
    what a SMILES writer wants and avoids hydrogens at bogus positions.
    """
    from rdkit import Chem

    props = mol.GetPropsAsDict()
    name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""

    # Strip any pre-existing Hs; any conformer on heavy atoms is preserved.
    mol_heavy = Chem.RemoveHs(mol)
    has_coords = mol_heavy.GetNumConformers() > 0

    # Apply Dimorphite-DL's pH-appropriate charges and H counts to the
    # molecule. Setting NoImplicit=True with an explicit H count makes
    # the atom state fully determined, which keeps kekulization happy on
    # aromatic heterocycles.
    new_states = _target_atom_states(mol_heavy, ph)
    mol_heavy = Chem.RWMol(mol_heavy)
    for idx, (charge, n_hs) in new_states.items():
        a = mol_heavy.GetAtomWithIdx(idx)
        a.SetFormalCharge(charge)
        a.SetNumExplicitHs(n_hs)
        a.SetNoImplicit(True)
    Chem.SanitizeMol(mol_heavy)

    # With 3D coordinates, add explicit hydrogens positioned from the
    # existing heavy-atom geometry (heavy-atom coordinates are not
    # modified). Without coordinates, or when the caller doesn't want
    # them, keep the protonation implicit so a SMILES writer renders it
    # cleanly without bogus zeroed positions.
    if has_coords and add_coord_hs:
        protonated = Chem.AddHs(mol_heavy, addCoords=True)
    else:
        protonated = mol_heavy

    # Restore name and SDF tags.
    if name:
        protonated.SetProp("_Name", name)
    for key, value in props.items():
        protonated.SetProp(key, str(value))
    return protonated


def protonate_smiles_string(smiles, ph=7.4):
    """
    Protonate a single SMILES string at `ph` and return the resulting
    SMILES. Convenience wrapper around `protonate_molecule` for the
    common string-in/string-out case (no coordinates involved).

    Raises ValueError if `smiles` cannot be parsed; other failures
    (e.g. Dimorphite-DL could not handle the molecule) propagate from
    `protonate_molecule`.
    """
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES {smiles!r}")
    protonated = protonate_molecule(mol, ph, add_coord_hs=False)
    return Chem.MolToSmiles(protonated)


def _is_smiles_path(path):
    return path.lower().endswith((".smi", ".smiles"))


def read_molecules(path):
    """
    Yield molecules from `path`, which may be SMILES (.smi/.smiles) or
    SDF. Unparseable entries are yielded as None so callers can count
    and report them. SMILES files are read as one molecule per line,
    "SMILES [optional name]".
    """
    from rdkit import Chem

    if _is_smiles_path(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                mol = Chem.MolFromSmiles(parts[0])
                if mol is not None and len(parts) > 1:
                    mol.SetProp("_Name", parts[1].strip())
                yield mol
    else:
        for mol in Chem.SDMolSupplier(path, removeHs=False, sanitize=True):
            yield mol


def make_writer(path):
    """Return an SDF or SMILES writer based on the output extension."""
    from rdkit import Chem

    if _is_smiles_path(path):
        return Chem.SmilesWriter(path, includeHeader=False)
    return Chem.SDWriter(path)


def protonate_ligands(input_path, output_path, ph):
    """Batch-protonate a ligand file (SDF/SMILES) into another."""
    writer = make_writer(output_path)
    # SMILES output never needs coordinate-bearing explicit hydrogens.
    add_coord_hs = not _is_smiles_path(output_path)

    n_in = n_out = n_fail = 0
    for mol in read_molecules(input_path):
        n_in += 1
        if mol is None:
            n_fail += 1
            print(
                f"[warn] skipping molecule {n_in}: RDKit failed to parse",
                file=sys.stderr,
            )
            continue
        try:
            protonated = protonate_molecule(mol, ph, add_coord_hs=add_coord_hs)
        except Exception as exc:
            n_fail += 1
            label = mol.GetProp("_Name") if mol.HasProp("_Name") else f"#{n_in}"
            print(f"[warn] skipping {label}: {exc}", file=sys.stderr)
            continue
        writer.write(protonated)
        n_out += 1

    writer.close()
    print(f"Read {n_in} molecules, wrote {n_out}, skipped {n_fail}.")


# ---------------------------------------------------------------------------
# Protein mode (Biotite + Hydride)
# ---------------------------------------------------------------------------

def reorder_hydrogens_after_heavy_atoms(atoms):
    """
    Return a new AtomArray where each hydrogen immediately follows the
    heavy atom it is bonded to.

    Hydrogens bonded to the same heavy atom appear in the order Hydride
    originally placed them. Orphan hydrogens (no heavy-atom bond found)
    are appended at the end as a fallback.

    The input AtomArray must have a populated BondList.
    """
    import numpy as np

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


def protonate_structure(structure, ligand_res_name=None, ph=7.0, relax=True):
    """
    Return a hydrogenated copy of a protein `AtomArray`.

    In-memory analogue of `protonate_molecule` for proteins: takes an
    AtomArray (e.g. from ``pdb.PDBFile.read(path).get_structure(model=1)``)
    and returns a new AtomArray with pH-appropriate hydrogens added and
    each hydrogen reordered to immediately follow its bonded heavy atom.

    If `ligand_res_name` is given (and not "none"), atoms with that
    residue name are removed first; a ValueError is raised if no such
    atoms exist. Any pre-existing hydrogens are stripped before Hydride
    adds them back.
    """
    import numpy as np
    import biotite.structure as struc
    import hydride

    # Optionally remove the ligand by residue name (3-letter CCD code).
    # "none" (any case) means "keep everything".
    if ligand_res_name is not None and ligand_res_name.lower() != "none":
        target = ligand_res_name.upper()
        keep_mask = np.char.upper(structure.res_name.astype(str)) != target
        if keep_mask.all():
            raise ValueError(
                f"No atoms with res_name '{target}' found in structure."
            )
        structure = structure[keep_mask]

    # Strip any pre-existing hydrogens; Hydride will add them itself.
    structure = structure[structure.element != "H"]

    # Assign covalent bonds from CCD residue templates.
    structure.bonds = struc.connect_via_residue_names(structure)

    # Set formal charges for canonical amino acids at the requested pH.
    charges = hydride.estimate_amino_acid_charges(structure, ph=ph)
    structure.set_annotation("charge", charges)

    # Add hydrogens, then optionally relax their geometry.
    structure, _ = hydride.add_hydrogen(structure)
    if relax:
        structure.coord = hydride.relax_hydrogen(structure)

    # Reorder so each hydrogen follows its bonded heavy atom.
    return reorder_hydrogens_after_heavy_atoms(structure)


def prepare_structure(input_path, ligand_res_name, output_path,
                      ph=7.0, relax=True, quiet=False):
    """
    Read a PDB file, protonate it with `protonate_structure`, and write
    the result to another PDB file. File-to-file driver analogous to
    `protonate_ligands` on the ligand side.
    """
    import biotite.structure.io.pdb as pdb

    structure = pdb.PDBFile.read(input_path).get_structure(model=1)
    structure = protonate_structure(
        structure, ligand_res_name=ligand_res_name, ph=ph, relax=relax
    )

    out = pdb.PDBFile()
    out.set_structure(structure)
    out.write(output_path)
    if not quiet:
        print(f"Wrote hydrogenated structure to {output_path}")


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode", required=True,
                           metavar="{ligand,protein}")

    lig = sub.add_parser(
        "ligand",
        help="Protonate small molecules (SDF/SMILES).",
        description="Protonate small molecules with Dimorphite-DL.",
    )
    lig.add_argument(
        "input", help="Path to the input file (SDF, or .smi/.smiles for SMILES)"
    )
    lig.add_argument(
        "output", help="Path to the output file (SDF, or .smi/.smiles for SMILES)"
    )
    lig.add_argument(
        "--ph", type=float, default=7.4,
        help="Target pH for protonation (default: 7.4)",
    )

    prot = sub.add_parser(
        "protein",
        help="Protonate a protein structure (PDB).",
        description="Add hydrogens to a protein with Hydride.",
    )
    prot.add_argument("input", help="Path to the input PDB file")
    prot.add_argument(
        "ligand_res_name",
        help="Residue name of the ligand to remove (e.g. ATP, HEM, AP5). "
             "Pass 'none' to skip ligand removal.",
    )
    prot.add_argument("output", help="Path to the output PDB file")
    prot.add_argument(
        "--ph", type=float, default=7.0,
        help="pH used to estimate amino-acid formal charges (default: 7.0)",
    )
    prot.add_argument(
        "--no-relax", action="store_true",
        help="Skip dihedral relaxation of hydrogens.",
    )

    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "ligand":
        protonate_ligands(args.input, args.output, args.ph)
    elif args.mode == "protein":
        prepare_structure(
            input_path=args.input,
            ligand_res_name=args.ligand_res_name,
            output_path=args.output,
            ph=args.ph,
            relax=not args.no_relax,
        )


if __name__ == "__main__":
    main()
