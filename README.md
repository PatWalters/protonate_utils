# protonate_utils

<p align="center">
  <img src="acid_robot.png" alt="A robot abstracting a proton from a carboxylic acid to leave a carboxylate" width="420">
</p>

A single utility for adding hydrogens to **ligands** and **proteins** at a
target pH, for use in molecular modeling and structure-based drug design.

## Why this exists

Most structures you download (a ligand from a database, a protein from the
PDB) are missing hydrogens, or carry hydrogens that don't reflect the
protonation state at physiological pH. Getting these right matters: a
carboxylic acid is deprotonated (`-COO⁻`) at pH 7.4, a basic amine is
protonated (`-NH₃⁺`), and a histidine side chain can go either way. Downstream
tasks (docking, free-energy calculations, MD simulations, electrostatics)
all depend on the correct charge and hydrogen placement.

Ligands and proteins need different tools for this. Small molecules are best
handled with cheminformatics pKa models; proteins need residue-aware logic and
geometry-based hydrogen placement. `protonate_utils.py` wraps the appropriate
specialist tool for each case behind one consistent interface, so you don't
have to remember two separate workflows:

- **Ligands** use [Dimorphite-DL](https://github.com/durrantlab/dimorphite_dl)
  for pH-aware protonation states and [the RDKit](https://www.rdkit.org/) for
  structure handling. When the input has 3D coordinates, the heavy-atom
  geometry is preserved exactly; only the newly added hydrogens are given
  computed positions.
- **Proteins** use [Hydride](https://hydride.biotite-python.org/) for
  geometry-based hydrogen addition and
  [Biotite](https://www.biotite-python.org/) for PDB handling, with formal
  charges estimated per amino acid at the requested pH.

Everything is exposed both as a **command-line tool** and as an importable
**Python API**.

## Installation

Install the latest release from [PyPI](https://pypi.org/project/protonate-utils/)
with `pip`:

```bash
pip install protonate-utils
```

Or install from a checkout for development:

```bash
git clone https://github.com/PatWalters/protonate_utils
cd protonate_utils
pip install -e .
```

Either way installs the dependencies for both modes (RDKit + Dimorphite-DL for
ligands, Biotite + Hydride + NumPy for proteins), puts a `protonate-utils`
command on your `PATH`, and makes `import protonate_utils` available.

## Command-line usage

Once installed, use the `protonate-utils` command. The first argument selects
the mode: `ligand` or `protein`. (You can also run it without installing via
`python protonate_utils.py …` from a checkout.)

### Ligands

```bash
# SDF in, SDF out (3D coordinates preserved, hydrogens placed from geometry)
protonate-utils ligand input.sdf output.sdf

# SMILES in, SMILES out, at a custom pH
protonate-utils ligand input.smi output.smi --ph 7.4

# Mixed: read SDF, write SMILES
protonate-utils ligand input.sdf output.smi
```

Input and output formats are inferred from the file extension:
`.smi`/`.smiles` is treated as SMILES, anything else as SDF. SMILES files are
read one molecule per line as `SMILES [optional name]`.

| Option   | Default | Description                          |
|----------|---------|--------------------------------------|
| `--ph`   | `7.4`   | Target pH for protonation.           |

Molecules that fail to parse or protonate are skipped with a warning on
stderr; the run reports how many were read, written, and skipped.

### Proteins

```bash
# Remove a bound ligand by residue name, then add hydrogens
protonate-utils protein input.pdb AP5 output.pdb

# Keep everything (no ligand removal)
protonate-utils protein input.pdb none output.pdb --ph 7.0
```

The second positional argument is the residue name (3-letter CCD code) of a
ligand to remove before protonation; pass `none` to keep all atoms. Output
hydrogens are reordered so each one immediately follows the heavy atom it is
bonded to.

| Option       | Default | Description                                         |
|--------------|---------|-----------------------------------------------------|
| `--ph`       | `7.0`   | pH used to estimate amino-acid formal charges.      |
| `--no-relax` | off     | Skip dihedral relaxation of the added hydrogens.    |

## Python API

Import the functions directly from `protonate_utils`. There are symmetric
in-memory and file-to-file entry points for both ligands and proteins.

|                  | Ligands                                  | Proteins                          |
|------------------|------------------------------------------|-----------------------------------|
| In-memory core   | `protonate_molecule(mol, ph)`            | `protonate_structure(structure, …)` |
| Convenience      | `protonate_smiles_string(smiles, ph)`    | N/A                               |
| File → file      | `protonate_ligands(in, out, ph)`         | `prepare_structure(in, res, out, …)` |
| I/O helpers      | `read_molecules(path)`, `make_writer(path)` | (Biotite `PDBFile`)            |

### Ligands

Protonate a single SMILES string and get a SMILES string back:

```python
from protonate_utils import protonate_smiles_string

protonate_smiles_string("CC(=O)O")             # 'CC(=O)[O-]'
protonate_smiles_string("OP(=O)(O)O", ph=7.4)  # 'O=P([O-])([O-])O'
```

`protonate_smiles_string` raises `ValueError` on an unparseable SMILES; other
failures (e.g. Dimorphite-DL cannot handle the molecule) propagate as
exceptions.

Protonate an RDKit `Mol` while preserving its 3D coordinates:

```python
from rdkit import Chem
from protonate_utils import protonate_molecule, read_molecules

mol = next(read_molecules("ligand.sdf"))
protonated = protonate_molecule(mol, ph=7.4)   # Mol with explicit Hs + coords
```

Pass `add_coord_hs=False` to keep protonation implicit (no explicit hydrogen
atoms added), appropriate when you intend to serialize to SMILES.

Batch-convert a whole file (the CLI ligand path):

```python
from protonate_utils import protonate_ligands

protonate_ligands("input.sdf", "output.sdf", ph=7.4)
```

### Proteins

Protonate an in-memory Biotite `AtomArray` and get a hydrogenated one back:

```python
import biotite.structure.io.pdb as pdb
from protonate_utils import protonate_structure

structure = pdb.PDBFile.read("input.pdb").get_structure(model=1)
hydrogenated = protonate_structure(
    structure,
    ligand_res_name="AP5",   # or None / "none" to keep all atoms
    ph=7.0,
    relax=True,
)
```

`protonate_structure` raises `ValueError` if `ligand_res_name` is given but no
atoms with that residue name exist. The returned `AtomArray` has hydrogens
added and reordered to follow their bonded heavy atoms.

Read a PDB, protonate, and write a PDB in one call (the CLI protein path):

```python
from protonate_utils import prepare_structure

prepare_structure("input.pdb", "AP5", "output.pdb", ph=7.0, relax=True)
```

## How it works

### Ligand protonation

1. Pre-existing hydrogens are stripped; any 3D conformer on the heavy atoms is
   kept.
2. Dimorphite-DL enumerates candidate microstate(s) within a ±0.5 pH window.
   One is chosen deterministically by a **site-by-site plausibility** check
   rather than by net charge (see
   [Correcting Dimorphite-DL microstates](#correcting-dimorphite-dl-microstates)
   below), and any residual implausible ionization is repaired against the
   input. The SMILES string is a final tiebreak, so re-runs are stable.
3. The chosen template's formal charges **and** total hydrogen counts are
   mapped back onto the original atoms via a charge-insensitive substructure
   match (so `-COOH` still matches `-COO⁻`). Carrying the H count, not just
   the charge, keeps the RDKit's kekulization correct on aromatic heterocycles.
4. With 3D input, `Chem.AddHs(addCoords=True)` adds hydrogens positioned from
   the existing geometry; heavy-atom coordinates are never moved. Without
   coordinates (SMILES), protonation stays implicit.

### Correcting Dimorphite-DL microstates

Dimorphite-DL enumerates *every* microstate whose modeled pKa falls anywhere
near the pH window, including many that are negligibly populated at pH 7.4. Left
to a "most ionized" or "closest net charge" rule, the selector picks chemically
wrong states: it deprotonates amides and phenols and protonates anilines. We add
a per-atom legitimacy check (`_charge_change_is_legitimate`) that compares each
candidate to the input atom-by-atom and accepts a formal-charge change only when
that group genuinely ionizes near physiological pH:

| Group | Typical pKa | At pH 7.4 | Dimorphite enumerates | We |
|-------|-------------|-----------|-----------------------|----|
| Aliphatic amine | pKaH ~10 | cation | both | **protonate** |
| Amidine / guanidine | pKaH ~12–13 | cation | both | **protonate** |
| Carboxylic acid | ~4 | anion | anion | **deprotonate** |
| Sulfonic / sulfinic / phosphate / phosphonate | <2–7 | anion | anion | **deprotonate** |
| Sulfonamide / acylsulfonamide / tetrazole | ~3–10 | anion | both | **deprotonate** |
| Carboxamide N–H | ~17–22 | neutral | both → `[N⁻]` *or* `[NH⁺]` | **keep neutral** |
| Aniline / amino-heteroarene | pKaH ~3–5 | neutral | both → `[NH⁺]` | **keep neutral** |
| Cyanamide (N–C≡N) | pKaH ~0 | neutral | both → `[NH⁺]` | **keep neutral** |
| Imidazole / pyrazole / indazole / indole / triazole N–H | ~10–17 | neutral | both → `[n⁻]` | **keep neutral** |
| Phenol / alcohol | ~10–16 | neutral | both → `[O⁻]` | **keep neutral** |
| Plain thiol / thione | ~7–10 | neutral | both → `[S⁻]` | **keep neutral** |

Two further safeguards:

- **Repair fallback.** When Dimorphite offers *only* an implausibly-ionized
  microstate (e.g. it returns just the `[N⁻]` form of an O-alkyl hydroxamate or
  imide, with no neutral alternative to select), the offending site is reverted
  to the input's protonation rather than emitted as-is.
- **Input charges preserved.** A change is only judged relative to the input, so
  charges already present in the SMILES (quaternary ammonium salts, *N*-oxides,
  mesoionic zwitterions) are never altered.

Borderline acids/bases whose pKa sits right at 7.4 (e.g. *p*-nitrophenol ~7.15,
mercaptoazoles ~7) are deliberately defaulted to neutral; they are ~50/50 at
physiological pH, so this is at least as defensible as ionizing them and avoids
mis-ionizing the far more common ordinary phenols and amides. Validated across
the 2,173-molecule Biogen logS set: no skips, no heavy-atom changes, and the
selection is deterministic.

### Protein protonation

1. Optionally remove a ligand by residue name, then strip any existing
   hydrogens.
2. Assign covalent bonds from CCD residue templates
   (`connect_via_residue_names`).
3. Estimate per-residue formal charges for canonical amino acids at the
   requested pH (`hydride.estimate_amino_acid_charges`).
4. Add hydrogens with Hydride and, by default, relax their geometry.
5. Reorder atoms so each hydrogen immediately follows the heavy atom it is
   bonded to.
