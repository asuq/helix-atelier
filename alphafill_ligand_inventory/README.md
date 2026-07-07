# AlphaFill Ligand Inventory

Create ligand and cofactor inventory tables from AlphaFill-enriched mmCIF
files.

The tool recursively scans AlphaFill CIF files, reads the `_entity`,
`_struct_asym`, `_chem_comp`, and `_atom_site` categories, and writes three TSV
files:

- `alphafill_ligand_inventory.tsv`: one row per detected ligand instance.
- `alphafill_model_summary.tsv`: one row per CIF/model.
- `alphafill_target_summary.tsv`: one row per inferred target/enzyme.

```bash
uv run alphafill_ligand_inventory \
    --input-root /path/to/01-4_alphafill \
    --outdir /path/to/01-4_alphafill/inventory
```

Useful options:

```bash
uv run alphafill_ligand_inventory \
    --input-root alphafill_outputs \
    --outdir alphafill_outputs/inventory \
    --pattern "*_alphafill.cif" \
    --top-model-only \
    --fail-on-empty
```

## Assumptions

- Input files are AlphaFill/AlphaFold-style mmCIF files, not arbitrary mmCIF
  archives.
- Target classes are inferred from path or filename tokens such as `mcr`,
  `ftr`, `mch`, `mer`, `mtd`, and `mtr`.
- AF3 model index is inferred from `_model_<number>` in the CIF filename.
- Water is excluded from the ligand inventory. Common buffers/additives are
  retained but marked as ignored.
- Cofactor classes are heuristic and project-oriented. High-priority hits still
  require structural inspection before biological interpretation.

## Limitations

This uses a small local mmCIF tokenizer and parser to avoid additional runtime
dependencies. It is adequate for the AlphaFill/AF3 output categories used here,
but it is not a complete mmCIF validator. For unusual CIF files, validate the
input independently and inspect warnings or unexpected empty summaries.
