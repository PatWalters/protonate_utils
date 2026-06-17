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
Read a local PDB file, optionally remove one or more ligands by residue
name (a comma-delimited list removes several at once, which also clears
buffer/ion residues), add hydrogens with Hydride at a target pH (default
7.0), and write the result to a PDB file. Hydrogens are reordered so that
each hydrogen immediately follows the heavy atom to which it is bonded.

Force-field protonation/tautomer residue names -- HID/HIE/HIP (and CHARMM
HSD/HSE/HSP), ASH, GLH, LYN, ARN, CYM, TYM, CYX -- are normalized to their
canonical CCD codes so bonds connect, and by default the exact state each
name encodes is preserved (e.g. HID keeps its proton on ND1, HIP stays the
+1 imidazolium, CYX sulfurs are disulfide-bonded). Pass --no-honor-protonation
to instead (re)protonate every residue purely from the target pH.

Install:
    pip install rdkit dimorphite-dl          # ligand mode
    pip install biotite hydride numpy        # protein mode

Usage:
    protonate_utils.py ligand input.sdf output.sdf
    protonate_utils.py ligand input.smi output.smi --ph 7.4
    protonate_utils.py protein input.pdb AP5 output.pdb
    protonate_utils.py protein input.pdb "EST,CL6" output.pdb  # remove several
    protonate_utils.py protein input.pdb none output.pdb --ph 7.0

The repository ships sample data: 1bmk_ligand.sdf and 1bmk_protein.pdb
(no bound ligand), plus 7axj_protein.pdb, which contains two pocket
ligands (EST and CL6) for exercising ligand removal:

    protonate_utils.py protein 7axj_protein.pdb "EST,CL6" 7axj_out.pdb
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


def _nitrogen_is_acylated_or_sulfonylated(n_atom):
    """
    True if `n_atom` is bonded to a carbonyl/thiocarbonyl carbon or a sulfonyl
    sulfur. Such a nitrogen (amide, imide, sulfonamide, N-acylsulfonamide) has
    its lone pair tied up by the adjacent electron-withdrawing group and is not
    basic, so it must never be *protonated* at physiological pH -- even the
    acidic acylsulfonamide/imide cases, which `_is_amide_nitrogen` deliberately
    excludes so their *deprotonation* stays allowed.
    """
    from rdkit import Chem

    for nbr in n_atom.GetNeighbors():
        z = nbr.GetAtomicNum()
        if z == 6:
            for b in nbr.GetBonds():
                other = b.GetOtherAtom(nbr)
                if (b.GetBondType() == Chem.BondType.DOUBLE
                        and other.GetAtomicNum() in (8, 16)):
                    return True
        elif z == 16:
            o_doubles = sum(
                1 for b in nbr.GetBonds()
                if b.GetBondType() == Chem.BondType.DOUBLE
                and b.GetOtherAtom(nbr).GetAtomicNum() == 8
            )
            if o_doubles >= 2:
                return True
    return False


def _is_aromatic_amine_nitrogen(n_atom):
    """
    True if `n_atom` is an aniline/aromatic-amine nitrogen -- a non-aromatic N
    bonded directly to an aromatic ring atom -- whose lone pair is delocalised
    into the ring. Such nitrogens are weak bases (aniline pKaH ~4.6;
    amino-pyridines/-pyrimidines/-azines pKaH ~3-5) and stay essentially neutral
    at pH 7.4, yet Dimorphite-DL still enumerates a protonated microstate for
    them.

    An *aliphatic* amine (no aromatic neighbour) and the C=N nitrogens of an
    amidine/guanidine/benzamidine (whose neighbouring carbon is not itself
    aromatic) are excluded here, so they remain protonatable. Strongly-basic
    amino-heteroarenes (e.g. 2-aminoimidazole, 4-aminopyridine) protonate on
    their *ring* nitrogen, a different atom, so this exclusion does not affect
    them.
    """
    if n_atom.GetAtomicNum() != 7 or n_atom.GetIsAromatic():
        return False
    if _is_amide_nitrogen(n_atom):
        return False
    return any(nbr.GetIsAromatic() for nbr in n_atom.GetNeighbors())


def _is_cyanamide_nitrogen(n_atom):
    """
    True if `n_atom` is a cyanamide nitrogen -- bonded to a nitrile carbon
    (N-C#N). The triple-bonded nitrile is strongly electron-withdrawing and
    ties up the nitrogen lone pair, so a dialkylcyanamide has pKaH ~0 (cyanamide
    itself is faintly *acidic*, pKa ~10) and is non-basic at pH 7.4. Dimorphite-DL
    nonetheless enumerates a protonated microstate for it, which we must reject.
    """
    from rdkit import Chem

    if n_atom.GetAtomicNum() != 7:
        return False
    for nbr in n_atom.GetNeighbors():
        if nbr.GetAtomicNum() != 6:
            continue
        for b in nbr.GetBonds():
            other = b.GetOtherAtom(nbr)
            if (b.GetBondType() == Chem.BondType.TRIPLE
                    and other.GetAtomicNum() == 7
                    and other.GetIdx() != n_atom.GetIdx()):
                return True
    return False


def _bonded_to_acidifying_centre(atom):
    """
    True if `atom` is bonded to an electron-withdrawing centre that makes an
    O-H/S-H on it a strong acid (pKa < ~7): a carbonyl/thiocarbonyl carbon
    (carboxyl/thioacid), a phosphorus oxyacid, or a sulfur oxyacid. Used to
    tell a genuine acid (carboxyl pKa ~4, sulfonic <2, phosphate ~1-7) apart
    from a weak one whose conjugate base is essentially absent at pH 7.4.
    """
    from rdkit import Chem

    for nbr in atom.GetNeighbors():
        z = nbr.GetAtomicNum()
        if z in (15, 16):
            # Phosphorus oxyacid, or sulfonic/sulfinic acid: the neighbouring
            # P/S bears at least one double-bonded oxygen.
            if any(
                b.GetBondType() == Chem.BondType.DOUBLE
                and b.GetOtherAtom(nbr).GetAtomicNum() == 8
                for b in nbr.GetBonds()
            ):
                return True
        elif z == 6:
            # Carbonyl/thiocarbonyl carbon -> carboxyl / thioacid.
            for b in nbr.GetBonds():
                other = b.GetOtherAtom(nbr)
                if (b.GetBondType() == Chem.BondType.DOUBLE
                        and other is not atom
                        and other.GetAtomicNum() in (8, 16)):
                    return True
    return False


def _is_acidic_oxygen(o_atom):
    """
    True if deprotonating this oxygen's O-H gives an anion that actually exists
    at pH 7.4 -- i.e. the oxygen of a carboxyl, sulfonic/sulfinic, or
    phosphorus oxyacid (pKa < ~7). A phenol (O on an aromatic carbon, pKa ~10),
    an alcohol (O on sp3 carbon, pKa ~16), or a hydroxy-heteroarene (really a
    neutral lactam tautomer) is >90% neutral at physiological pH, yet
    Dimorphite-DL still enumerates its ``[O-]`` microstate, so those must be
    rejected -- the acid-side analogue of the weak amide/azole N-H.
    """
    return o_atom.GetAtomicNum() == 8 and _bonded_to_acidifying_centre(o_atom)


def _is_acidic_sulfur(s_atom):
    """
    True if deprotonating this sulfur's S-H gives a thiolate present at pH 7.4
    -- a thioacid S adjacent to a carbonyl (pKa ~3) or a sulfur/phosphorus
    oxyacid. A plain alkyl thiol (pKa ~10.5) or aromatic thiol/thione (pKa ~7,
    e.g. a mercaptoazole) is predominantly neutral, so its Dimorphite-enumerated
    ``[S-]`` microstate is rejected.
    """
    return s_atom.GetAtomicNum() == 16 and _bonded_to_acidifying_centre(s_atom)


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
    amidine, guanidine, aromatic N). Deprotonation to an anion is sensible on a
    strong oxygen/sulfur acid (carboxyl, sulfonic/sulfinic, phosphate, thioacid)
    and on a genuinely acidic nitrogen (sulfonamide, tetrazole, ...). It is
    *not* sensible on the weakly-acidic groups that Dimorphite-DL nonetheless
    enumerates a deprotonated microstate for: a plain carboxamide (pKa ~17-22)
    or aromatic N-H heterocycle (imidazole/pyrazole/indazole/indole, pKa
    ~13-17), nor a phenol (pKa ~10), alcohol (pKa ~16), or plain thiol/thione
    (pKa ~7-10), all >90% neutral at pH 7.4. Flagging those here lets the
    selector reject them.
    """
    if delta_q > 0:
        # Protonation to a cation. Only a nitrogen base accepts a proton near
        # physiological pH. An amide nitrogen is *not* basic (its conjugate
        # acid pKa is ~0), so reject protonation there even though Dimorphite-DL
        # enumerates the [NH+] microstate.
        if atom.GetAtomicNum() != 7:
            return False
        if _nitrogen_is_acylated_or_sulfonylated(atom):
            return False
        if _is_aromatic_amine_nitrogen(atom):
            return False
        if _is_cyanamide_nitrogen(atom):
            return False
        return True
    # delta_q < 0: deprotonation to an anion.
    z = atom.GetAtomicNum()
    if z == 8:
        # Carboxyl/sulfonate/phosphate oxygen deprotonates; phenol/alcohol
        # (pKa ~10-16) stays neutral at pH 7.4.
        return _is_acidic_oxygen(atom)
    if z == 16:
        # Thioacid sulfur deprotonates; plain thiol/thione stays neutral.
        return _is_acidic_sulfur(atom)
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


def _repair_illegitimate_ionizations(input_mol, cand_smiles):
    """
    Revert any still-illegitimate ionization in `cand_smiles` to the input's
    protonation at that atom.

    Selection (`_pick_state`) can only choose among the microstates
    Dimorphite-DL offers. For an activated-but-not-acidic nitrogen -- an
    O-alkyl hydroxamate, an acylhydrazide, a plain imide -- Dimorphite may
    return *only* the deprotonated ``[N-]`` form, with no neutral alternative
    to pick. Here we align the chosen candidate to the input atom-by-atom and,
    for every formal-charge change that isn't a legitimate ionization (see
    `_charge_change_is_legitimate`), copy the input atom's charge and hydrogen
    count back onto the candidate. Genuine acids handled correctly upstream
    (carboxyl, sulfonamide, tetrazole, acylsulfonamide) have *legitimate*
    changes and are left untouched.

    Returns a canonical SMILES, or `cand_smiles` unchanged if the molecules
    can't be aligned or the repaired structure won't sanitize.
    """
    from rdkit import Chem

    if input_mol is None:
        return cand_smiles
    cand_mol = Chem.MolFromSmiles(cand_smiles)
    if cand_mol is None:
        return cand_smiles

    match = _skeleton_copy(input_mol).GetSubstructMatch(_skeleton_copy(cand_mol))
    if not match or len(match) != cand_mol.GetNumAtoms():
        return cand_smiles

    rw = Chem.RWMol(cand_mol)
    changed = False
    for cand_idx, input_idx in enumerate(match):
        ca = rw.GetAtomWithIdx(cand_idx)
        ia = input_mol.GetAtomWithIdx(input_idx)
        delta_q = ca.GetFormalCharge() - ia.GetFormalCharge()
        if delta_q and not _charge_change_is_legitimate(ca, delta_q):
            ca.SetFormalCharge(ia.GetFormalCharge())
            ca.SetNumExplicitHs(ia.GetTotalNumHs())
            ca.SetNoImplicit(True)
            changed = True

    if not changed:
        return cand_smiles
    try:
        repaired = rw.GetMol()
        Chem.SanitizeMol(repaired)
    except Exception:
        return cand_smiles
    return Chem.MolToSmiles(repaired)


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

    best = min(states, key=score)
    # The best available state may still carry an illegitimate ionization when
    # Dimorphite offered no cleaner alternative; revert those sites to the input.
    return _repair_illegitimate_ionizations(input_mol, best)


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

    When `add_coord_hs` is set (the SDF-output path), explicit hydrogens
    are added so they appear in the written file. If the input carries a
    3D conformer they are positioned from the existing geometry while the
    heavy-atom coordinates are preserved; without coordinates (SMILES
    input) they are still added explicitly, just without positions.
    Otherwise (`add_coord_hs` False, the SMILES-output path) protonation
    is left implicit, which is what a SMILES writer wants and avoids
    hydrogens at bogus positions.
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

    # For SDF output, add explicit hydrogens so they are written to the
    # file. With 3D coordinates they are positioned from the existing
    # heavy-atom geometry (heavy-atom coordinates are not modified); with
    # none (SMILES input) they are added without coordinates. For SMILES
    # output the caller passes add_coord_hs=False, keeping protonation
    # implicit so the SMILES writer renders it cleanly.
    if add_coord_hs:
        protonated = Chem.AddHs(mol_heavy, addCoords=has_coords)
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


def _looks_like_smiles_header(line):
    """
    True if `line` is a column header (e.g. "SMILES Name") rather than a
    molecule record -- i.e. its first whitespace-delimited token does not
    parse as a SMILES. RDKit's error logging is silenced during the probe
    so the expected parse failure doesn't print a spurious error.
    """
    from rdkit import Chem, RDLogger

    token = line.split(None, 1)[0]
    RDLogger.DisableLog("rdApp.error")
    try:
        return Chem.MolFromSmiles(token) is None
    finally:
        RDLogger.EnableLog("rdApp.error")


def read_molecules(path):
    """
    Yield molecules from `path`, which may be SMILES (.smi/.smiles) or
    SDF. Unparseable entries are yielded as None so callers can count
    and report them. SMILES files are read as one molecule per line,
    "SMILES [optional name]"; an optional leading header line (e.g.
    "SMILES Name"), recognized by its first token not parsing as a
    SMILES, is skipped.
    """
    from rdkit import Chem

    if _is_smiles_path(path):
        with open(path) as fh:
            first = True
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if first:
                    first = False
                    # A header line isn't a molecule record; skip it
                    # silently rather than reporting a parse failure.
                    if _looks_like_smiles_header(line):
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

# AMBER/CHARMM force-field protonation-state residue names mapped to their
# canonical CCD amino-acid code. These variants either are absent from the CCD
# or -- worse -- collide with unrelated CCD components (e.g. "HIE" and "HID"
# are registered as completely different small molecules, not histidine
# tautomers), so connect_via_residue_names matches against the wrong template
# and assigns zero bonds. They all share their parent residue's heavy-atom
# connectivity, and we strip and re-add hydrogens at the requested pH anyway,
# so renaming to the standard code before bonding is safe and lets Hydride
# place hydrogens correctly.
_PROTONATION_RESNAME_ALIASES = {
    # Histidine tautomers/protonation states (AMBER HIx, CHARMM HSx).
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
    "HSD": "HIS", "HSE": "HIS", "HSP": "HIS",
    # Cysteine: disulfide-bonded / deprotonated thiolate.
    "CYX": "CYS", "CYM": "CYS",
    # Neutral (protonated) aspartate / glutamate.
    "ASH": "ASP", "GLH": "GLU",
    # Neutral lysine, deprotonated tyrosine, neutral arginine.
    "LYN": "LYS", "TYM": "TYR", "ARN": "ARG",
}


def _normalize_protonation_resnames(structure):
    """
    Rewrite force-field protonation-state residue names (HID, HIE, CYX, ...)
    in-place to their canonical CCD amino-acid codes so that
    ``connect_via_residue_names`` can assign covalent bonds. Returns the
    structure for convenience. See `_PROTONATION_RESNAME_ALIASES`.
    """
    import numpy as np

    res_name = np.char.upper(structure.res_name.astype(str))
    for alias, canonical in _PROTONATION_RESNAME_ALIASES.items():
        res_name[res_name == alias] = canonical
    structure.res_name = res_name
    return structure


# Per-atom formal-charge overrides that pin a residue to the protonation state
# its force-field name encodes, overriding Hydride's pH-based estimate. Applied
# only when `honor_protonation` is set. The HID/HIE tautomer difference is *not*
# a charge difference (both are neutral) and is handled separately by swapping
# imidazole ring bond orders; HIP is the +1 imidazolium and is pinned here.
_PROTONATION_CHARGE_OVERRIDES = {
    "HIP": {"ND1": 1}, "HSP": {"ND1": 1},          # imidazolium, +1
    "HID": {"ND1": 0, "NE2": 0}, "HSD": {"ND1": 0, "NE2": 0},  # neutral
    "HIE": {"ND1": 0, "NE2": 0}, "HSE": {"ND1": 0, "NE2": 0},  # neutral
    "ASH": {"OD1": 0, "OD2": 0},                   # neutral aspartic acid
    "GLH": {"OE1": 0, "OE2": 0},                   # neutral glutamic acid
    "LYN": {"NZ": 0},                              # neutral lysine
    "ARN": {"NH1": 0, "NH2": 0},                   # neutral arginine
    "CYM": {"SG": -1},                             # cysteine thiolate
    "TYM": {"OH": -1},                             # tyrosinate
    "CYX": {"SG": 0},                              # disulfide cysteine (+SS bond)
}

# Force-field names for the delta-protonated histidine tautomer (H on ND1).
# The CCD HIS template is the epsilon tautomer (H on NE2), so these need the
# imidazole ring double bond moved from ND1=CE1 to CE1=NE2.
_DELTA_HISTIDINE_RESNAMES = {"HID", "HSD"}


def _swap_to_delta_histidine(bonds, name_to_idx):
    """
    Move the imidazole ring double bond so the added hydrogen lands on ND1
    (delta tautomer) instead of NE2. No-op if any ring atom is missing.
    """
    import biotite.structure as struc

    nd1 = name_to_idx.get("ND1")
    ce1 = name_to_idx.get("CE1")
    ne2 = name_to_idx.get("NE2")
    if nd1 is None or ce1 is None or ne2 is None:
        return
    bonds.remove_bond(nd1, ce1)
    bonds.add_bond(nd1, ce1, struc.BondType.AROMATIC_SINGLE)
    bonds.remove_bond(ce1, ne2)
    bonds.add_bond(ce1, ne2, struc.BondType.AROMATIC_DOUBLE)


def _bond_disulfides(structure, sg_indices, cutoff=2.5):
    """
    Add an S-S single bond between each disulfide-cysteine SG and its nearest
    partner SG within `cutoff` angstrom, so Hydride leaves those sulfurs
    unprotonated. Each sulfur is paired at most once.
    """
    import numpy as np
    import biotite.structure as struc

    coord = structure.coord
    bonded = set()
    for a in sg_indices:
        if a in bonded:
            continue
        best, best_d = None, cutoff
        for b in sg_indices:
            if b == a or b in bonded:
                continue
            d = float(np.linalg.norm(coord[a] - coord[b]))
            if d < best_d:
                best, best_d = b, d
        if best is not None:
            structure.bonds.add_bond(a, best, struc.BondType.SINGLE)
            bonded.add(a)
            bonded.add(best)


def _enforce_input_protonation(structure, original_res_names, charges):
    """
    Pin each residue to the exact protonation/tautomer state encoded by its
    original force-field name, instead of letting Hydride re-decide from pH.

    `charges` (Hydride's pH estimate, aligned to `structure`) is overridden in
    place and returned; HID/HSD residues additionally get their imidazole ring
    bond orders swapped so the hydrogen lands on ND1, and CYX pairs get an
    explicit S-S bond. `structure.bonds` is modified in place. `original_res_names`
    holds the residue names as they were *before* normalization to CCD codes.
    """
    import biotite.structure as struc

    starts = struc.get_residue_starts(structure, add_exclusive_stop=True)
    cyx_sg = []
    for k in range(len(starts) - 1):
        start, stop = int(starts[k]), int(starts[k + 1])
        variant = str(original_res_names[start])
        overrides = _PROTONATION_CHARGE_OVERRIDES.get(variant)
        if overrides is None:
            continue
        name_to_idx = {
            str(structure.atom_name[i]): i for i in range(start, stop)
        }
        for atom_name, q in overrides.items():
            i = name_to_idx.get(atom_name)
            if i is not None:
                charges[i] = q
        if variant in _DELTA_HISTIDINE_RESNAMES:
            _swap_to_delta_histidine(structure.bonds, name_to_idx)
        if variant == "CYX":
            sg = name_to_idx.get("SG")
            if sg is not None:
                cyx_sg.append(sg)
    _bond_disulfides(structure, cyx_sg)
    return charges


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


def protonate_structure(structure, ligand_res_name=None, ph=7.0, relax=True,
                        honor_protonation=True):
    """
    Return a hydrogenated copy of a protein `AtomArray`.

    In-memory analogue of `protonate_molecule` for proteins: takes an
    AtomArray (e.g. from ``pdb.PDBFile.read(path).get_structure(model=1)``)
    and returns a new AtomArray with pH-appropriate hydrogens added and
    each hydrogen reordered to immediately follow its bonded heavy atom.

    If `ligand_res_name` is given (and not "none"), atoms with that
    residue name are removed first. Several residues may be removed at
    once by passing a comma-delimited list (e.g. "EST,CL6"); a ValueError
    is raised naming any residue name not present in the structure. Any
    pre-existing hydrogens are stripped before Hydride adds them back.

    When `honor_protonation` is True (the default), residues named with a
    force-field protonation/tautomer code -- HID/HIE/HIP (and CHARMM
    HSD/HSE/HSP), ASH, GLH, LYN, ARN, CYM, TYM, CYX -- are pinned to exactly
    the state that name encodes, overriding Hydride's pH estimate (e.g. HID
    keeps its proton on ND1, HIP stays the +1 imidazolium, CYX sulfurs are
    S-S bonded and left unprotonated). With it False, every residue is
    (re)protonated purely from `ph`, so the input HID/HIE/... distinction is
    discarded.
    """
    import numpy as np
    import biotite.structure as struc
    import hydride

    # Optionally remove one or more ligands by residue name (3-letter CCD
    # code). A comma-delimited list removes several at once, e.g. "EST,CL6"
    # to clear both ligands (and any buffer/ion residues) from a pocket.
    # "none" (any case) means "keep everything".
    if ligand_res_name is not None and ligand_res_name.lower() != "none":
        targets = [
            name.strip().upper()
            for name in ligand_res_name.split(",")
            if name.strip()
        ]
        upper_res = np.char.upper(structure.res_name.astype(str))
        missing = [t for t in targets if not (upper_res == t).any()]
        if missing:
            raise ValueError(
                "No atoms with res_name "
                + ", ".join(repr(m) for m in missing)
                + " found in structure."
            )
        structure = structure[~np.isin(upper_res, targets)]

    # Strip any pre-existing hydrogens; Hydride will add them itself.
    structure = structure[structure.element != "H"]

    # Capture the force-field protonation names before they are normalized
    # away, so we can re-impose the exact states they encode further down.
    original_res_names = np.char.upper(structure.res_name.astype(str))

    # Normalize force-field protonation-state residue names (HID/HIE/CYX/...)
    # to canonical CCD codes; otherwise connect_via_residue_names matches them
    # against the wrong template (or none) and leaves those residues unbonded.
    structure = _normalize_protonation_resnames(structure)

    # Assign covalent bonds from CCD residue templates.
    structure.bonds = struc.connect_via_residue_names(structure)

    # Set formal charges for canonical amino acids at the requested pH,
    # then optionally pin the force-field-named residues to their encoded state.
    charges = hydride.estimate_amino_acid_charges(structure, ph=ph)
    if honor_protonation:
        charges = _enforce_input_protonation(
            structure, original_res_names, charges
        )
    structure.set_annotation("charge", charges)

    # Add hydrogens, then optionally relax their geometry.
    structure, _ = hydride.add_hydrogen(structure)
    if relax:
        structure.coord = hydride.relax_hydrogen(structure)

    # Reorder so each hydrogen follows its bonded heavy atom.
    return reorder_hydrogens_after_heavy_atoms(structure)


def prepare_structure(input_path, ligand_res_name, output_path,
                      ph=7.0, relax=True, honor_protonation=True, quiet=False):
    """
    Read a PDB file, protonate it with `protonate_structure`, and write
    the result to another PDB file. File-to-file driver analogous to
    `protonate_ligands` on the ligand side.
    """
    import biotite.structure.io.pdb as pdb

    structure = pdb.PDBFile.read(input_path).get_structure(model=1)
    structure = protonate_structure(
        structure, ligand_res_name=ligand_res_name, ph=ph, relax=relax,
        honor_protonation=honor_protonation,
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
             "Pass a comma-delimited list to remove several at once "
             "(e.g. 'EST,CL6'), or 'none' to skip ligand removal.",
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
    prot.add_argument(
        "--no-honor-protonation", action="store_true",
        help="Ignore force-field protonation names (HID/HIE/HIP, ASH, GLH, "
             "CYX, ...) and (re)protonate every residue from --ph instead.",
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
            honor_protonation=not args.no_honor_protonation,
        )


if __name__ == "__main__":
    main()
