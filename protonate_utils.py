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

def _skeleton_copy(mol):
    """
    Return a charge- and H-agnostic copy of `mol` for use as a
    substructure-match template that aligns two molecules differing only in
    protonation state (an input molecule against a Dimorphite-DL microstate of
    itself).

    Bond orders and aromaticity are preserved -- they are what distinguishes,
    say, an amidine's ``=N`` from its ``-N``, so flattening them would let the
    charge/H mapping land on the wrong nitrogen and blow up its valence.
    Only formal charges, explicit Hs and radicals are cleared, and implicit
    Hs are switched off so a neutralized cation can't overflow its valence.

    Crucially we do *not* run a full sanitize: re-kekulizing a neutralized
    aromatic cation such as a protonated pyridinium ``[nH+]`` is what made the
    indazole/imidazole molecules fail. The molecule keeps the aromaticity
    perceived at parse time, and a light property-cache/ring refresh is all
    substructure matching needs, so this never raises.
    """
    from rdkit import Chem

    m = Chem.RWMol(mol)
    for a in m.GetAtoms():
        a.SetFormalCharge(0)
        a.SetNumExplicitHs(0)
        a.SetNoImplicit(True)
        a.SetNumRadicalElectrons(0)
    m.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(m)
    return m


def _is_amide_nitrogen(n_atom):
    """
    True if `n_atom` is a (thio)carboxamide nitrogen -- bonded to a carbon
    that bears a double bond to O or S -- and *not* also bonded to a sulfonyl
    group. A plain carboxamide N-H has pKa ~17-22 and stays neutral at
    physiological pH, but an (acyl)sulfonamide N-H is genuinely acidic, so we
    exclude that case (the caller treats its deprotonation as legitimate).
    """
    from rdkit import Chem

    has_carbonyl = False
    has_sulfonyl = False
    for nbr in n_atom.GetNeighbors():
        z = nbr.GetAtomicNum()
        if z == 6:
            for b in nbr.GetBonds():
                other = b.GetOtherAtom(nbr)
                if (b.GetBondType() == Chem.BondType.DOUBLE
                        and other.GetAtomicNum() in (8, 16)):
                    has_carbonyl = True
        elif z == 16:
            # Sulfonyl S(=O)(=O) neighbour -> acidic (acyl)sulfonamide.
            o_doubles = sum(
                1 for b in nbr.GetBonds()
                if b.GetBondType() == Chem.BondType.DOUBLE
                and b.GetOtherAtom(nbr).GetAtomicNum() == 8
            )
            if o_doubles >= 2:
                has_sulfonyl = True
    return has_carbonyl and not has_sulfonyl


def _is_acidic_aromatic_nitrogen(n_atom):
    """
    True if `n_atom` is an aromatic ring N-H acidic enough to deprotonate near
    physiological pH. In practice that means only tetrazole-grade azoles,
    whose ring carries four nitrogens (N-H pKa ~4.9). The common aromatic
    N-H heterocycles -- pyrrole/indole (1 ring N, pKa ~17), imidazole/pyrazole
    (2 N, pKa ~14), triazole (3 N, pKa ~10) -- are >99% neutral at pH 7.4, so
    their Dimorphite-enumerated ``[n-]`` microstates must be rejected.
    """
    mol = n_atom.GetOwningMol()
    ring_info = mol.GetRingInfo()
    idx = n_atom.GetIdx()
    most_ring_nitrogens = 0
    for ring in ring_info.AtomRings():
        if idx in ring:
            n_count = sum(
                1 for i in ring if mol.GetAtomWithIdx(i).GetAtomicNum() == 7
            )
            most_ring_nitrogens = max(most_ring_nitrogens, n_count)
    return most_ring_nitrogens >= 4


def _charge_change_is_legitimate(atom, delta_q):
    """
    Decide whether changing `atom`'s formal charge by `delta_q` (candidate
    minus input) reflects a real ionization near physiological pH.

    Protonation to a cation is only sensible on a nitrogen base (amine,
    amidine, guanidine, aromatic N). Deprotonation to an anion is sensible on
    an oxygen/sulfur acid (carboxyl, phenol, thiol, phosphate) and on a
    genuinely acidic nitrogen (sulfonamide, tetrazole, ...). It is *not*
    sensible on the weakly-acidic nitrogen groups that Dimorphite-DL
    nonetheless enumerates a deprotonated microstate for: a plain carboxamide
    (pKa ~17-22) or an ordinary aromatic N-H heterocycle such as
    imidazole/pyrazole/indazole/indole (pKa ~13-17). Flagging those here lets
    the selector reject them.
    """
    if delta_q > 0:
        return atom.GetAtomicNum() == 7
    # delta_q < 0: deprotonation to an anion.
    z = atom.GetAtomicNum()
    if z in (8, 16):
        return True
    if z == 7:
        if _is_amide_nitrogen(atom):
            return False
        if atom.GetIsAromatic() and not _is_acidic_aromatic_nitrogen(atom):
            return False
        return True
    return False


def _count_illegitimate_ionizations(input_mol, cand_mol):
    """
    Align `cand_mol` to `input_mol` atom-by-atom -- their heavy-atom
    skeletons are identical, only protonation differs -- and count the formal
    charge changes that don't correspond to a legitimate ionization (see
    `_charge_change_is_legitimate`). Comparing against the input (rather than
    against neutral) means a charge already present in the input is never
    penalised; only newly introduced, chemically implausible ionizations are.

    Returns a large sentinel if the two can't be aligned, so such candidates
    sort last without crashing the selection.
    """
    match = _skeleton_copy(input_mol).GetSubstructMatch(
        _skeleton_copy(cand_mol)
    )
    if not match or len(match) != cand_mol.GetNumAtoms():
        return 1_000_000

    bad = 0
    for cand_idx, input_idx in enumerate(match):
        ca = cand_mol.GetAtomWithIdx(cand_idx)
        ia = input_mol.GetAtomWithIdx(input_idx)
        delta_q = ca.GetFormalCharge() - ia.GetFormalCharge()
        if delta_q and not _charge_change_is_legitimate(ca, delta_q):
            bad += 1
    return bad


def _pick_state(input_smiles, states):
    """
    Choose one microstate from Dimorphite-DL's output deterministically.

    Dimorphite returns every microstate whose pKa falls within the
    requested pH window. Its list order is not stable across Python
    runs, so taking `states[0]` makes the pipeline non-deterministic:
    e.g. a secondary alkyl amide can come back as either NH or N-, and
    we'd silently flip between them on re-runs.

    Selection happens in two tiers (lower is better):

    1. **Site-by-site plausibility.** Each candidate is aligned to the
       input atom-by-atom and its formal-charge changes are checked: a
       cation must form on a nitrogen base, an anion on an O/S acid or a
       genuinely acidic nitrogen (sulfonamide, tetrazole, ...). Dimorphite
       also enumerates implausible microstates -- most notably a
       deprotonated carboxamide ``C(=O)[N-]`` (N-H pKa ~17-22) -- and those
       are penalised by their count of illegitimate changes, so the neutral
       amide is kept over its bogus anion.

    2. **Most ionized.** Among equally plausible candidates, prefer the one
       with the greatest total ionic character (sum of |formal charge|).
       When dimorphite is unsure it returns both the ionized and the neutral
       form (a primary amine comes back as both ``CCC[NH3+]`` and ``CCCN``);
       this keeps the ionized one and, unlike "match the input charge", does
       not collapse back to a neutral input drawn without explicit charges. A
       zwitterion (net charge 0 but two charged atoms) is preferred over its
       neutral form.

    The SMILES string is a final deterministic tiebreak. For groups with
    pKa far from the pH window dimorphite returns a single state, so the
    choice only matters when dimorphite is unsure.
    """
    from rdkit import Chem

    input_mol = Chem.MolFromSmiles(input_smiles)

    def score(smi):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            # Unparseable candidate: sort strictly last.
            return (1_000_000, 0, smi)
        illegitimate = (
            _count_illegitimate_ionizations(input_mol, m)
            if input_mol is not None else 0
        )
        ionic = sum(abs(a.GetFormalCharge()) for a in m.GetAtoms())
        # Fewest implausible ionizations first, then most ionized
        # (negate for min), then SMILES tiebreak.
        return (illegitimate, -ionic, smi)

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

    # Map heavy-atom indices between original and template via a skeleton
    # (charge/H/bond-order-agnostic) match, so e.g. -COOH still matches -COO-
    # and protonated aromatic heterocycles still match their neutral form.
    match = _skeleton_copy(mol_heavy).GetSubstructMatch(
        _skeleton_copy(template)
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
