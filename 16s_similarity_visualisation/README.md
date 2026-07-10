# MEGA 16S similarity visualisation

`make_16s_similarity_heatmap.py` reads a MEGA/MEGA-CC lower-left
p-distance matrix, converts the distances to 16S rRNA gene sequence
similarity percentages, writes reusable tabular outputs, and draws a heatmap
with optional hierarchical clustering and dendrograms. Taxon labels come
directly from the MEGA file; no metadata table is required.

## Method and interpretation

MEGA p-distance is the fraction of compared nucleotide sites that differ. The
script calculates similarity as:

```text
similarity_percent = 100 * (1 - p_distance)
```

The default 98.65% value is a commonly used 16S rRNA gene species-screening
reference. It is not a definitive species boundary or a substitute for genomic,
phylogenetic, phenotypic, or ecological evidence. The reference is:

> Kim M, Oh H-S, Park S-C, Chun J. Towards a taxonomic coherence between
> average nucleotide identity and 16S rRNA gene sequence similarity for species
> demarcation of prokaryotes. International Journal of Systematic and
> Evolutionary Microbiology. 2014;64:346-351.
> DOI: [10.1099/ijs.0.059774-0](https://doi.org/10.1099/ijs.0.059774-0).

The script visualises the distances reported by MEGA and does not recalculate
alignments or effective comparison lengths. With pairwise deletion, different
taxon pairs can be evaluated over different sets or numbers of sites; this can
affect comparability and should be considered when interpreting the heatmap.

## Installation

Python 3.12 or newer and [uv](https://docs.astral.sh/uv/) are recommended:

```bash
cd 16s_similarity_visualisation
uv sync
```

The runtime dependencies are NumPy, pandas, SciPy, and Matplotlib. The locked
environment in `uv.lock` records the exact resolved dependency versions.

## Usage

```bash
uv run python make_16s_similarity_heatmap.py \
  --mega-distance /tmp/16S_pdistance.meg \
  --out-prefix results/16S_similarity \
  --threshold 98.65 \
  --label-mode compact \
  --fig-format svg
```

Run `uv run python make_16s_similarity_heatmap.py --help` for all options.
Clustering uses the p-distance matrix with average linkage by default. The
accepted linkage methods (`single`, `complete`, `average`, and `weighted`) can
operate on precomputed dissimilarities. `--no-cluster` preserves the original
MEGA order.

Label modes are:

- `original`: retain the MEGA label after removal of its leading `#`.
- `pretty`: replace underscores with spaces.
- `compact`: preserve recognised accessions and make trailing organism or
  strain text readable.

## Outputs

For `--out-prefix results/16S_similarity`, the script writes:

- `16S_similarity.pdistance.tsv`: full square p-distance matrix.
- `16S_similarity.similarity_percent.tsv`: full square similarity matrix.
- `16S_similarity.below_threshold.tsv`: full square Boolean threshold mask.
- `16S_similarity.pairwise_long.tsv`: both directed off-diagonal records for
  every pair.
- `16S_similarity.clustered_order.txt`: original taxon identifiers in plotted
  order.
- `16S_similarity.heatmap.svg`: clustered heatmap in the selected format.

Threshold comparisons use unrounded values and are strictly below (`<`) the
selected value. The cell grid is always black; when values are annotated,
below-threshold off-diagonal values are red. Diagonal values remain black.
Summary logs count unique unordered biological pairs and directed matrix cells
separately.

## Tests

```bash
uv run python -m pytest
```
