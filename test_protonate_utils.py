#!/usr/bin/env python
"""
Regression tests for ligand protonation-state selection.

These cover the `_pick_state` microstate choice and the end-to-end
`protonate_smiles_string` result at physiological pH. The motivating
bug: when Dimorphite-DL returns both the ionized and the neutral
microstate for an uncertain group (e.g. a primary amine returns both
``CCC[NH3+]`` and ``CCCN``), the selector must keep the *most ionized*
state rather than collapsing back to the neutral input.

Runs under pytest, or standalone: ``python test_protonate_utils.py``.
Requires the ligand extra: ``pip install rdkit dimorphite-dl``.
"""

from rdkit import Chem

from protonate_utils import _pick_state, protonate_smiles_string


def _canon(smiles):
    """Canonical SMILES for order-independent comparison."""
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def test_pick_state_prefers_ionized_amine():
    # Dimorphite returns both microstates for a primary amine; the
    # protonated (+1) form must win over the neutral input form.
    assert _canon(_pick_state("CCCN", ["CCC[NH3+]", "CCCN"])) == _canon("CCC[NH3+]")


def test_pick_state_prefers_zwitterion_over_neutral():
    # Net charge is 0 for both, but the zwitterion has two charged atoms
    # (more total ionic character) and must be preferred.
    states = ["[NH3+]CC(=O)[O-]", "NCC(=O)O"]
    assert _canon(_pick_state("NCC(=O)O", states)) == _canon("[NH3+]CC(=O)[O-]")


def test_pick_state_is_deterministic_under_reordering():
    # List order from Dimorphite is not stable across runs; the choice
    # must not depend on it.
    a = _pick_state("CCCN", ["CCC[NH3+]", "CCCN"])
    b = _pick_state("CCCN", ["CCCN", "CCC[NH3+]"])
    assert _canon(a) == _canon(b)


def test_amine_is_protonated_end_to_end():
    assert _canon(protonate_smiles_string("CCCN", ph=7.4)) == _canon("CCC[NH3+]")


def test_carboxylic_acid_is_deprotonated_end_to_end():
    assert _canon(protonate_smiles_string("CCC(=O)O", ph=7.4)) == _canon("CCC(=O)[O-]")


def test_glycine_is_zwitterion_end_to_end():
    assert _canon(protonate_smiles_string("NCC(=O)O", ph=7.4)) == _canon(
        "[NH3+]CC(=O)[O-]"
    )


def test_pick_state_keeps_amide_neutral():
    # Dimorphite enumerates a deprotonated carboxamide microstate, but an
    # amide N-H (pKa ~17-22) is neutral at pH 7.4. The "most ionized" rule
    # alone would wrongly pick the [N-]; the site-by-site check must reject
    # it and keep the neutral amide.
    states = ["CC(=O)[N-]C", "CC(=O)NC"]
    assert _canon(_pick_state("CC(=O)NC", states)) == _canon("CC(=O)NC")


def test_pick_state_amide_neutral_amine_protonated():
    # A molecule with both an amide and a basic amine: the amide stays
    # neutral while the amine is protonated.
    states = [
        "CC(=O)NCC[NH3+]",  # legitimate: amine protonated
        "CC(=O)[N-]CCN",    # illegitimate: amide deprotonated
        "CC(=O)NCCN",       # nothing ionized
    ]
    assert _canon(_pick_state("CC(=O)NCCN", states)) == _canon("CC(=O)NCC[NH3+]")


def test_amide_not_deprotonated_end_to_end():
    # Regression for the reported bug: a secondary amide must not come back
    # as its deprotonated [N-] form.
    out = _canon(protonate_smiles_string("CC(=O)NC1CC1", ph=7.4))
    assert "-" not in out, f"amide was deprotonated: {out}"
    assert out == _canon("CC(=O)NC1CC1")


def test_pick_state_keeps_ordinary_azole_neutral():
    # An imidazole N-H (pKa ~14.5) is neutral at pH 7.4; the [n-] microstate
    # Dimorphite enumerates must be rejected, not chosen as "most ionized".
    assert _canon(_pick_state("c1cnc[nH]1", ["c1cnc[n-]1", "c1cnc[nH]1"])) == _canon(
        "c1cnc[nH]1"
    )


def test_pick_state_deprotonates_tetrazole():
    # A tetrazole N-H (pKa ~4.9) IS acidic at pH 7.4, so the anion is correct.
    assert _canon(_pick_state("c1nnn[nH]1", ["c1nnn[n-]1", "c1nnn[nH]1"])) == _canon(
        "c1nnn[n-]1"
    )


def test_aromatic_heterocycle_does_not_crash_and_stays_neutral():
    # Regression: neutralizing a protonated aromatic heterocycle used to make
    # RDKit fail to kekulize, so indazole/imidazole molecules were skipped.
    # They must now process and keep the weakly-acidic ring N-H neutral.
    smi = "CC(C)(C)c1cc(NC(=O)c2cccc3cn[nH]c23)[nH]n1"  # fused indazole + pyrazole
    out = _canon(protonate_smiles_string(smi, ph=7.4))
    assert "[n-]" not in out, f"azole was deprotonated: {out}"
    assert out == _canon(smi)


def test_guanidine_is_protonated_end_to_end():
    # Regression: bond orders must be preserved during atom mapping so an
    # amidine/guanidine =N is not confused with -N (which overflowed valence
    # and skipped the molecule). The strong base must end up protonated.
    out = _canon(protonate_smiles_string("N=C(N)N/N=C/c1c(Cl)cccc1Cl", ph=7.4))
    assert "+" in out, f"guanidine was not protonated: {out}"
    assert out == _canon("NC(=[NH2+])N/N=C/c1c(Cl)cccc1Cl")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
