"""Regression tests: basic side chains stay protonated in non-canonical peptides.

Peptides with several backbone amides drive Dimorphite-DL's microstate enumeration
into net-negative (amide-deprotonation) combinations, so it never offers the
amine-protonated microstate and the basic side chain comes out neutral -- the net
charge is then too low (see the repair-up pass, `_repair_missing_protonations`).

These 10 cases each embed a well-known **non-canonical amino acid** (public chemistry:
ornithine, 2,4-diaminobutyric acid, 2,3-diaminopropionic acid, N-alpha-methyl-lysine,
norleucine, citrulline, alpha-aminoisobutyric acid, sarcosine) in an Ac-...-NH2 peptide
with a basic centre, and assert the correct net charge. Without repair-up every one of
these returns a net charge one (or two) too low; the neutral non-canonicals (Nle, Cit,
Aib, Sar) also confirm the repair does NOT over-protonate a urea/amide or a plain chain.
"""
import pytest
from rdkit import Chem

from protonate_utils import protonate_smiles_string


def _net_charge(smiles):
    m = Chem.MolFromSmiles(smiles)
    assert m is not None, f"unparseable output SMILES: {smiles!r}"
    return Chem.GetFormalCharge(m)


# (name, input SMILES [Ac-...-NH2, both termini neutral], expected net charge at pH 7.4)
# net = (# basic side chains) - (# acidic side chains); each peptide has >=6 backbone
# amides, enough to exhaust Dimorphite's variant budget with amide-anion combinations.
NONCANONICAL_BASIC_CASES = [
    ("ornithine",
     "CC(=O)N[C@@H](C)C(=O)NCC(=O)N[C@@H](CCCN)C(=O)N[C@@H](C)C(=O)NCC(=O)N[C@@H](C)C(N)=O", 1),
    ("2,4-diaminobutyric acid (Dab)",
     "CC(=O)NCC(=O)N[C@@H](C)C(=O)N[C@@H](CCN)C(=O)NCC(=O)N[C@@H](C)C(=O)N[C@@H](C)C(N)=O", 1),
    ("2,3-diaminopropionic acid (Dap)",
     "CC(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)N[C@@H](CN)C(=O)NCC(=O)NCC(=O)N[C@@H](C)C(N)=O", 1),
    ("N-alpha-methyl-lysine",
     "CC(=O)N[C@@H](C)C(=O)N(C)[C@@H](CCCCN)C(=O)N[C@@H](C)C(=O)NCC(=O)N[C@@H](C)C(=O)N[C@@H](C)C(N)=O", 1),
    ("two ornithines",
     "CC(=O)N[C@@H](CCCN)C(=O)N[C@@H](C)C(=O)NCC(=O)N[C@@H](C)C(=O)N[C@@H](CCCN)C(=O)N[C@@H](C)C(N)=O", 2),
    ("norleucine + lysine",
     "CCCC[C@H](NC(C)=O)C(=O)N[C@@H](C)C(=O)N[C@@H](CCCCN)C(=O)NCC(=O)N[C@@H](C)C(=O)N[C@@H](C)C(N)=O", 1),
    ("citrulline + lysine (urea must stay neutral)",
     "CC(=O)N[C@@H](CCCNC(N)=O)C(=O)N[C@@H](C)C(=O)N[C@@H](CCCCN)C(=O)NCC(=O)N[C@@H](C)C(=O)N[C@@H](C)C(N)=O", 1),
    ("alpha-aminoisobutyric acid (Aib) + lysine",
     "CC(=O)NC(C)(C)C(=O)N[C@@H](C)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](C)C(=O)NCC(=O)N[C@@H](C)C(N)=O", 1),
    ("sarcosine + lysine",
     "CC(=O)N(C)CC(=O)N[C@@H](C)C(=O)N[C@@H](CCCCN)C(=O)NCC(=O)N[C@@H](C)C(=O)N[C@@H](C)C(N)=O", 1),
    ("ornithine + glutamate (base + acid)",
     "CC(=O)N[C@@H](CCCN)C(=O)N[C@@H](C)C(=O)N[C@@H](CCC(=O)O)C(=O)NCC(=O)N[C@@H](C)C(=O)N[C@@H](C)C(N)=O", 0),
]


@pytest.mark.parametrize("name,smiles,expected", NONCANONICAL_BASIC_CASES,
                         ids=[c[0] for c in NONCANONICAL_BASIC_CASES])
def test_noncanonical_peptide_net_charge(name, smiles, expected):
    out = protonate_smiles_string(smiles, ph=7.4)
    got = _net_charge(out)
    assert got == expected, f"{name}: net {got:+d}, expected {expected:+d}  ->  {out}"


if __name__ == "__main__":
    passed = 0
    for name, smiles, expected in NONCANONICAL_BASIC_CASES:
        got = _net_charge(protonate_smiles_string(smiles, ph=7.4))
        ok = got == expected
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: net {got:+d} (expected {expected:+d})")
    print(f"\n{passed}/{len(NONCANONICAL_BASIC_CASES)} passed")
