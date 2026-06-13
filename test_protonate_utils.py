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


def test_pick_state_keeps_amide_unprotonated():
    # Dimorphite enumerates a *protonated* amide microstate [NH+], but an amide
    # nitrogen is not basic (conjugate-acid pKa ~0), so it must stay neutral.
    # The "most ionized" rule alone would wrongly pick the [NH+]; the
    # site-by-site check must reject the protonation.
    states = ["CC(=O)[NH+]1CCCCC1", "CC(=O)N1CCCCC1"]
    assert _canon(_pick_state("CC(=O)N1CCCCC1", states)) == _canon("CC(=O)N1CCCCC1")


def test_tertiary_amide_not_protonated_end_to_end():
    # Regression for the reported bug: an acrylamide on a piperidine came back
    # with the ring amide N protonated to [NH+]. It must stay neutral.
    smi = "C=CC(=O)N1C[C@H](Nc2ncnc3[nH]ccc23)CC[C@@H]1C"
    out = _canon(protonate_smiles_string(smi, ph=7.4))
    amide_n = Chem.MolFromSmarts("[NX3;+0][CX3]=[OX1]")
    assert Chem.MolFromSmiles(out).HasSubstructMatch(amide_n), (
        f"amide nitrogen was protonated: {out}"
    )


def test_phenol_is_not_deprotonated_end_to_end():
    # A phenol (pKa ~10) is >99% neutral at pH 7.4, but Dimorphite enumerates
    # the phenolate. The "most ionized" rule would wrongly pick [O-]; the
    # site-by-site acid check must keep the neutral phenol.
    out = _canon(protonate_smiles_string("c1ccccc1O", ph=7.4))
    assert "[O-]" not in out, f"phenol was deprotonated: {out}"
    assert out == _canon("c1ccccc1O")


def test_pick_state_keeps_phenol_neutral():
    assert _canon(_pick_state("c1ccccc1O", ["[O-]c1ccccc1", "Oc1ccccc1"])) == _canon(
        "Oc1ccccc1"
    )


def test_alcohol_is_not_deprotonated_end_to_end():
    # An aliphatic alcohol (pKa ~16) must never ionize at physiological pH.
    out = _canon(protonate_smiles_string("OCCc1ccccc1", ph=7.4))
    assert "[O-]" not in out, f"alcohol was deprotonated: {out}"


def test_thiophenol_is_not_deprotonated_end_to_end():
    out = _canon(protonate_smiles_string("c1ccccc1S", ph=7.4))
    assert "[S-]" not in out, f"thiophenol was deprotonated: {out}"


def test_carboxyl_ionizes_but_phenols_do_not():
    # Regression for the net -5 outlier: of a molecule with one carboxylic acid
    # (pKa ~4) and several catechol phenols (pKa ~10), only the carboxyl ionizes.
    smi = "O=C(O)c1cc2cc(O)c(O)cc2c(C(=O)c2ccc(O)c(O)c2)n1"
    out = protonate_smiles_string(smi, ph=7.4)
    m = Chem.MolFromSmiles(out)
    net = sum(a.GetFormalCharge() for a in m.GetAtoms())
    assert net == -1, f"expected net -1 (carboxylate only), got {net}: {out}"


def test_sulfonic_and_phosphonic_acids_still_deprotonate():
    # The acid-side tightening must not touch genuine strong O-acids.
    assert "[O-]" in protonate_smiles_string("CCS(=O)(=O)O", ph=7.4)
    assert "[O-]" in protonate_smiles_string("CCP(=O)(O)O", ph=7.4)


def test_hydroxamate_single_state_repaired_to_neutral():
    # Dimorphite returns ONLY the deprotonated [N-] microstate for an O-alkyl
    # hydroxamate (pKa ~8-9, predominantly neutral at 7.4) -- there is no
    # neutral state to select. The per-site repair must revert it to neutral.
    out = _canon(protonate_smiles_string("CCONC(=O)c1cncn1-c1ccc(F)cc1", ph=7.4))
    assert "[N-]" not in out, f"hydroxamate was left deprotonated: {out}"
    assert out == _canon("CCONC(=O)c1cncn1-c1ccc(F)cc1")


def test_imide_single_state_repaired_to_neutral():
    # A cyclic imide N (between two carbonyls, pKa ~8-9) likewise comes back
    # only as the anion; repair keeps the predominant neutral form.
    smi = "COc1cc(C=C2C(=O)NN(c3ccccc3)C2=O)ccc1OCC(N)=O"
    out = _canon(protonate_smiles_string(smi, ph=7.4))
    assert "[N-]" not in out, f"imide was left deprotonated: {out}"


def test_acylsulfonamide_is_deprotonated_end_to_end():
    # The repair must NOT touch a genuine acid: an N-acylsulfonamide (pKa ~3-5)
    # is correctly anionic at pH 7.4 and stays deprotonated.
    out = _canon(protonate_smiles_string("Cc1ccc(S(=O)(=O)NC(=O)c2ccccc2)cc1", ph=7.4))
    assert "[N-]" in out, f"acylsulfonamide was not deprotonated: {out}"


def test_acylsulfonamide_not_protonated_end_to_end():
    # A cyclic N-acylsulfonamide (N flanked by C=O and SO2) is non-basic and
    # weakly acidic, never protonated. _is_amide_nitrogen excludes it (to allow
    # deprotonation), so the broader acyl/sulfonyl check must block protonation.
    smi = "Cc1nc(-c2ccccc2)ccc1C(=O)N1CCCS1(=O)=O"
    out = _canon(protonate_smiles_string(smi, ph=7.4))
    assert "+" not in out, f"acylsulfonamide N was protonated: {out}"
    assert out == _canon(smi)


def test_aniline_is_not_protonated_end_to_end():
    # Aniline (pKaH ~4.6) is a weak base and stays neutral at pH 7.4, even
    # though Dimorphite enumerates the anilinium microstate.
    out = _canon(protonate_smiles_string("c1ccccc1N", ph=7.4))
    assert "+" not in out, f"aniline was protonated: {out}"
    assert out == _canon("c1ccccc1N")


def test_aminoheteroarene_amine_stays_neutral_end_to_end():
    # The exocyclic amine on the reported deazapurine is an aromatic amine
    # (pKaH ~3-5) and must stay neutral; only aliphatic/amidine N protonates.
    smi = "C=CC(=O)N1C[C@H](Nc2ncnc3[nH]ccc23)CC[C@@H]1C"
    out = _canon(protonate_smiles_string(smi, ph=7.4))
    assert "+" not in out, f"aromatic amine was protonated: {out}"
    assert out == _canon(smi)


def test_benzylamine_still_protonates_end_to_end():
    # A benzylic amine is aliphatic (N on sp3 carbon, not aromatic), a genuine
    # base -- the aromatic-amine rejection must not catch it.
    out = _canon(protonate_smiles_string("c1ccccc1CN", ph=7.4))
    assert _canon(out) == _canon("[NH3+]Cc1ccccc1")


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
