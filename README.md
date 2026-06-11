# protonate_utils

A single utility for adding hydrogens to **ligands** and **proteins** at a
target pH, for use in molecular modeling and structure-based drug design.

## Why this exists

Most structures you download — a ligand from a database, a protein from the
PDB — are missing hydrogens, or carry hydrogens that don't reflect the
protonation state at physiological pH. Getting these right matters: a
carboxylic acid is deprotonated (`-COO⁻`) at pH 7.4, a basic amine is
protonated (`-NH₃⁺`), and a histidine side chain can go either way. Downstream
tasks — docking, free-energy calculations, MD simulations, electrostatics —
all depend on the correct charge and hydrogen placement.

Ligands and proteins need different tools for this. Small molecules are best
handled with cheminformatics pKa models; proteins need residue-aware logic and
geometry-based hydrogen placement. `protonate_utils.py` wraps the appropriate
specialist tool for each case behind one consistent interface, so you don't
have to remember two separate workflows:

- **Ligands** use [Dimorphite-DL](https://github.com/durrantlab/dimorphite_dl)
  for pH-aware protonation states and [RDKit](https://www.rdkit.org/) for
  structure handling. When the input has 3D coordinates, the heavy-atom
  geometry is preserved exactly — only the newly added hydrogens are given
  computed positions.
- **Proteins** use [Hydride](https://hydride.biotite-python.org/) for
  geometry-based hydrogen addition and
  [Biotite](https://www.biotite-python.org/) for PDB handling, with formal
  charges estimated per amino acid at the requested pH.

Everything is exposed both as a **command-line tool** and as an importable
**Python API**.

## Installation

The two modes have independent dependencies; install whichever you need
(imports are lazy, so ligand mode never requires the protein libraries and
vice versa).

```bash
# Ligand mode
pip install rdkit dimorphite-dl

# Protein mode
pip install biotite hydride numpy
```

## Command-line usage

The first argument selects the mode: `ligand` or `protein`.

### Ligands

```bash
# SDF in, SDF out (3D coordinates preserved, hydrogens placed from geometry)
python protonate_utils.py ligand input.sdf output.sdf

# SMILES in, SMILES out, at a custom pH
python protonate_utils.py ligand input.smi output.smi --ph 7.4

# Mixed: read SDF, write SMILES
python protonate_utils.py ligand input.sdf output.smi
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
python protonate_utils.py protein input.pdb AP5 output.pdb

# Keep everything (no ligand removal)
python protonate_utils.py protein input.pdb none output.pdb --ph 7.0
```

The second positional argument is the residue name (3-letter CCD code) of a
ligand to remove before protonation — pass `none` to keep all atoms. Output
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
| Convenience      | `protonate_smiles_string(smiles, ph)`    | —                                 |
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
atoms added) — appropriate when you intend to serialize to SMILES.

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
2. Dimorphite-DL predicts the dominant microstate(s) within a ±0.5 pH window.
   When it returns more than one, the state whose net formal charge is closest
   to the input molecule's charge is chosen deterministically (with the SMILES
   string as a tiebreak), so re-runs are stable.
3. The chosen template's formal charges **and** total hydrogen counts are
   mapped back onto the original atoms via a charge-insensitive substructure
   match (so `-COOH` still matches `-COO⁻`). Carrying the H count — not just
   the charge — keeps RDKit's kekulization correct on aromatic heterocycles.
4. With 3D input, `Chem.AddHs(addCoords=True)` adds hydrogens positioned from
   the existing geometry; heavy-atom coordinates are never moved. Without
   coordinates (SMILES), protonation stays implicit.

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
