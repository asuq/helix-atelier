# AlphaFill Ligand Inventory

Create ligand and cofactor inventory tables from AlphaFill-enriched mmCIF
files and matching AlphaFill JSON metadata.

The tool recursively scans AlphaFill CIF files, reads the `_entity`,
`_struct_asym`, `_chem_comp`, and `_atom_site` categories, pairs each CIF with
its same-stem AlphaFill JSON metadata file when available, and writes TSV
files:

- `alphafill_ligand_inventory.tsv`: one row per detected ligand instance.
- `alphafill_model_summary.tsv`: one row per CIF/model.
- `alphafill_target_summary.tsv`: one row per inferred target/enzyme.
- `alphafill_json_transplants.tsv`: one row per JSON donor/transplant record.
- `alphafill_donor_summary.tsv`: one row per target/model/ligand/donor
  combination.

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
    --json-pattern "*_alphafill.json" \
    --close-mcr-donors 9QR1,9QQT,9QR3,9QM5 \
    --top-model-only \
    --fail-on-empty
```

JSON parsing is enabled by default. Use `--no-json` to disable it, and
`--warn-missing-json` to print warnings when a CIF has no matching JSON
metadata file.

## Assumptions

- Input files are AlphaFill/AlphaFold-style mmCIF files, not arbitrary mmCIF
  archives.
- JSON metadata files are model-specific and normally share the CIF stem, for
  example `sample_model_0_alphafill.cif` and
  `sample_model_0_alphafill.json`.
- Target classes are inferred from path or filename tokens such as `mcr`,
  `ftr`, `mch`, `mer`, `mtd`, and `mtr`.
- AF3 model index is inferred from `_model_<number>` in the CIF filename.
- Water is excluded from the ligand inventory. Common buffers/additives are
  retained but marked as ignored.
- JSON `analogue_id` is preferred over donor `compound_id` for
  `ligand_comp_id`, because it better matches the ligand code placed into the
  AlphaFill-enriched model.
- Cofactor classes are heuristic and project-oriented. High-priority hits still
  require structural inspection before biological interpretation.
- Long file path columns are written at the end of TSV tables.

## Limitations

This uses a small local mmCIF tokenizer and parser to avoid additional runtime
dependencies. It is adequate for the AlphaFill/AF3 output categories used here,
but it is not a complete mmCIF validator. For unusual CIF files, validate the
input independently and inspect warnings or unexpected empty summaries.

The JSON parser is schema-tolerant and recursively searches for likely
transplant records. It is designed for AlphaFill `hits[].transplants[]`
metadata, but missing JSON files and unfamiliar optional fields are non-fatal.
