"""
Add hydrogens to molecules in an SDF file at a target pH (default 7.4),
preserving the original 3D coordinates of the heavy atoms.

pH-aware protonation states are predicted with Dimorphite-DL, the
resulting formal charges *and* H counts are mapped back onto the
original 3D molecule via substructure matching, and explicit hydrogens
are added with Chem.AddHs(addCoords=True). Heavy-atom positions are not
disturbed; only the newly added hydrogens get computed positions based
on the existing geometry.

Install:
    pip install rdkit dimorphite-dl

Usage:
    python protonate_sdf.py input.sdf output.sdf
    python protonate_sdf.py input.sdf output.sdf --ph 7.4
"""

import argparse
import sys

from rdkit import Chem
from dimorphite_dl import protonate_smiles


def _neutralized_copy(mol):
    """
    Return a copy of `mol` with all formal charges and explicit H counts
    zeroed, for use as a charge-insensitive substructure-match template.
    Sanitization is best-effort; if neutralization produces an invalid
    valence, we fall back to a partial sanitize that still updates ring
    info (which substructure matching uses).
    """
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


def protonate_molecule(mol, ph):
    """
    Return a Mol with pH-appropriate protonation and explicit Hs added,
    while preserving the 3D coordinates of the input heavy atoms.
    """
    props = mol.GetPropsAsDict()
    name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""

    # Strip any pre-existing Hs; conformer on heavy atoms is preserved.
    mol_heavy = Chem.RemoveHs(mol)
    if mol_heavy.GetNumConformers() == 0:
        raise RuntimeError("Input molecule has no 3D coordinates")

    # Apply Dimorphite-DL's pH-appropriate charges and H counts to the
    # 3D molecule. Setting NoImplicit=True with an explicit H count makes
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

    # Add explicit hydrogens with positions derived from existing
    # heavy-atom geometry. Heavy-atom coordinates are not modified.
    protonated = Chem.AddHs(mol_heavy, addCoords=True)

    # Restore name and SDF tags.
    if name:
        protonated.SetProp("_Name", name)
    for key, value in props.items():
        protonated.SetProp(key, str(value))
    return protonated


def main(input_path, output_path, ph):
    suppl = Chem.SDMolSupplier(input_path, removeHs=False, sanitize=True)
    writer = Chem.SDWriter(output_path)

    n_in = n_out = n_fail = 0
    for mol in suppl:
        n_in += 1
        if mol is None:
            n_fail += 1
            print(
                f"[warn] skipping molecule {n_in}: RDKit failed to parse",
                file=sys.stderr,
            )
            continue
        try:
            protonated = protonate_molecule(mol, ph)
        except Exception as exc:
            n_fail += 1
            label = mol.GetProp("_Name") if mol.HasProp("_Name") else f"#{n_in}"
            print(f"[warn] skipping {label}: {exc}", file=sys.stderr)
            continue
        writer.write(protonated)
        n_out += 1

    writer.close()
    print(f"Read {n_in} molecules, wrote {n_out}, skipped {n_fail}.")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_sdf", help="Path to the input SDF")
    p.add_argument("output_sdf", help="Path to the output SDF")
    p.add_argument(
        "--ph", type=float, default=7.4,
        help="Target pH for protonation (default: 7.4)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.input_sdf, args.output_sdf, args.ph)
