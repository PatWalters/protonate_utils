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
