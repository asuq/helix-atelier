# AF3 Input Prep

Prepare AlphaFold 3 local-codebase or AlphaFold Server JSON input files from
small TSV specifications.

```bash
uv run af3_input_prep single input.tsv --style local --outdir af3_inputs
uv run af3_input_prep batch job1.tsv job2.tsv --style server --outdir af3_inputs --prefix screen
```

Run `uv run af3_input_prep --help` for the TSV schema and command options.
