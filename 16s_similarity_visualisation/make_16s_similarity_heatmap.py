#!/usr/bin/env python3
"""Convert a MEGA p-distance matrix to a 16S similarity heatmap."""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


LOGGER = logging.getLogger(__name__)
DEFAULT_THRESHOLD = 98.65
AUTO_ANNOTATION_MAX_TAXA = 40
CLUSTER_METHODS = ("single", "complete", "average", "weighted")
FIGURE_FORMATS = ("pdf", "png", "svg")
LABEL_MODES = ("original", "pretty", "compact")
ANNOTATION_MODES = ("true", "false", "auto")
INDEXED_LINE_PATTERN = re.compile(r"^\s*\[\s*(\d+)\s*\]\s*(.*?)\s*$")
GENOME_ACCESSION_PATTERN = re.compile(
    r"^(GC[AF]_\d+(?:\.\d+)?_\d+)(?:_(.+))?$"
)
NUCLEOTIDE_ACCESSION_PATTERN = re.compile(
    r"^([A-Z]{1,4}_\d+(?:\.\d+)?)(?:_(.+))?$"
)


@dataclass(frozen=True)
class MegaDistanceData:
    """Parsed MEGA taxon labels and a full symmetric p-distance matrix."""

    labels: list[str]
    p_distance: np.ndarray


@dataclass(frozen=True)
class OutputPaths:
    """All files produced for one output prefix."""

    p_distance: Path
    similarity: Path
    below_threshold: Path
    pairwise_long: Path
    clustered_order: Path
    heatmap: Path


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert a MEGA/MEGA-CC lower-left p-distance matrix to 16S "
            "rRNA gene sequence similarities and a clustered heatmap."
        )
    )
    parser.add_argument(
        "--mega-distance",
        required=True,
        type=Path,
        help="MEGA lower-left p-distance matrix file.",
    )
    parser.add_argument(
        "--out-prefix",
        required=True,
        type=Path,
        help="Output path prefix; parent directories are created if needed.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="16S similarity screening threshold in percent. Default: 98.65.",
    )
    parser.add_argument(
        "--fig-format",
        choices=FIGURE_FORMATS,
        default="pdf",
        help="Heatmap format. Default: pdf.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure resolution for raster output. Default: 300.",
    )
    parser.add_argument(
        "--cluster-method",
        choices=CLUSTER_METHODS,
        default="average",
        help="Hierarchical linkage method. Default: average.",
    )
    parser.add_argument(
        "--no-cluster",
        action="store_true",
        help="Retain the original MEGA order and omit dendrograms.",
    )
    parser.add_argument(
        "--label-mode",
        choices=LABEL_MODES,
        default="compact",
        help="Plot-label transformation. Default: compact.",
    )
    parser.add_argument(
        "--max-label-length",
        type=int,
        default=90,
        help="Maximum plot-label length, including trailing '...'. Default: 90.",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        help="Figure width in inches; otherwise derived from the taxon count.",
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        help="Figure height in inches; otherwise derived from the taxon count.",
    )
    parser.add_argument(
        "--annotate-values",
        choices=ANNOTATION_MODES,
        default="auto",
        help="Cell annotations: true, false, or auto (taxa <= 40). Default: auto.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.threshold) or not 0.0 <= args.threshold <= 100.0:
        parser.error("--threshold must be finite and within [0, 100]")
    if args.dpi <= 0:
        parser.error("--dpi must be greater than zero")
    if args.max_label_length < 4:
        parser.error("--max-label-length must be at least 4")
    for option_name in ("fig_width", "fig_height"):
        value = getattr(args, option_name)
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            parser.error(f"--{option_name.replace('_', '-')} must be finite and positive")
    return args


def parse_mega_distance(path: Path) -> MegaDistanceData:
    """Parse and strictly validate a MEGA lower-left distance matrix."""
    if not path.is_file():
        raise ValueError(f"MEGA distance file not found: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"Could not read MEGA distance file {path}: {error}") from error

    ntaxa_values: list[int] = []
    labels_by_index: dict[int, str] = {}
    matrix_rows: dict[int, tuple[int, str]] = {}

    for line_number, line in enumerate(lines, start=1):
        if line.lstrip().lower().startswith("!format"):
            match = re.search(r"\bNTaxa\s*=\s*(\d+)\b", line, re.IGNORECASE)
            if match is None:
                raise ValueError(f"Line {line_number}: !Format line lacks NTaxa")
            ntaxa_values.append(int(match.group(1)))

        indexed_match = INDEXED_LINE_PATTERN.match(line)
        if indexed_match is None:
            continue
        index = int(indexed_match.group(1))
        content = indexed_match.group(2).strip()
        if content.startswith("#"):
            label = content[1:].strip()
            if not label:
                raise ValueError(f"Line {line_number}: taxon {index} has an empty label")
            if index in labels_by_index:
                raise ValueError(f"Line {line_number}: duplicate taxon-label index {index}")
            labels_by_index[index] = label
        else:
            if index in matrix_rows:
                raise ValueError(f"Line {line_number}: duplicate matrix-row index {index}")
            matrix_rows[index] = (line_number, content)

    if not ntaxa_values:
        raise ValueError("No !Format line with NTaxa was found")
    if len(set(ntaxa_values)) != 1:
        raise ValueError(f"Conflicting NTaxa values found: {ntaxa_values}")
    ntaxa = ntaxa_values[0]
    if ntaxa < 2:
        raise ValueError(f"NTaxa must be at least 2, got {ntaxa}")

    expected_indices = set(range(1, ntaxa + 1))
    _validate_indices("taxon-label", labels_by_index, expected_indices)
    _validate_indices("matrix-row", matrix_rows, expected_indices)

    labels = [labels_by_index[index] for index in range(1, ntaxa + 1)]
    duplicate_labels = sorted(
        {label for label in labels if labels.count(label) > 1}
    )
    if duplicate_labels:
        raise ValueError("Duplicate taxon label(s): " + ", ".join(duplicate_labels))

    matrix = np.zeros((ntaxa, ntaxa), dtype=np.float64)
    for one_based_index in range(1, ntaxa + 1):
        line_number, content = matrix_rows[one_based_index]
        tokens = content.split()
        expected_count = one_based_index - 1
        if len(tokens) != expected_count:
            raise ValueError(
                f"Line {line_number}: matrix row {one_based_index} expected "
                f"{expected_count} value(s), found {len(tokens)}"
            )
        for column_index, token in enumerate(tokens):
            try:
                value = float(token)
            except ValueError as error:
                raise ValueError(
                    f"Line {line_number}: non-numeric p-distance at matrix row "
                    f"{one_based_index}, column {column_index + 1}: {token!r}"
                ) from error
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"Line {line_number}: p-distance outside [0, 1] at matrix "
                    f"row {one_based_index}, column {column_index + 1}: {token!r}"
                )
            row_index = one_based_index - 1
            matrix[row_index, column_index] = value
            matrix[column_index, row_index] = value

    validate_p_distance_matrix(matrix, ntaxa)
    return MegaDistanceData(labels=labels, p_distance=matrix)


def _validate_indices(
    description: str,
    indexed_values: dict[int, object],
    expected_indices: set[int],
) -> None:
    """Require indexed MEGA records to cover exactly 1 through NTaxa."""
    observed_indices = set(indexed_values)
    missing = sorted(expected_indices - observed_indices)
    unexpected = sorted(observed_indices - expected_indices)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(map(str, missing)))
        if unexpected:
            details.append("unexpected " + ", ".join(map(str, unexpected)))
        raise ValueError(f"Invalid {description} indices: {'; '.join(details)}")


def validate_p_distance_matrix(matrix: np.ndarray, ntaxa: int) -> None:
    """Validate dimensions, range, diagonal, and symmetry of p-distances."""
    if matrix.shape != (ntaxa, ntaxa):
        raise ValueError(
            f"P-distance matrix shape must be {(ntaxa, ntaxa)}, got {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("P-distance matrix contains non-finite values")
    if np.any((matrix < 0.0) | (matrix > 1.0)):
        raise ValueError("P-distance matrix contains values outside [0, 1]")
    if not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=0.0):
        raise ValueError("P-distance matrix diagonal must be 0.0")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise ValueError("P-distance matrix is not symmetric")


def calculate_similarity(p_distance: np.ndarray) -> np.ndarray:
    """Convert p-distance fractions to 16S similarity percentages."""
    similarity = 100.0 * (1.0 - p_distance)
    np.fill_diagonal(similarity, 100.0)
    if not np.all(np.isfinite(similarity)):
        raise ValueError("Similarity matrix contains non-finite values")
    if np.any((similarity < 0.0) | (similarity > 100.0)):
        raise ValueError("Similarity matrix contains values outside [0, 100]")
    if not np.allclose(similarity, similarity.T, rtol=0.0, atol=1e-10):
        raise ValueError("Similarity matrix is not symmetric")
    return similarity


def calculate_below_threshold(
    similarity: np.ndarray, threshold: float
) -> np.ndarray:
    """Return a symmetric strict-below-threshold mask with a false diagonal."""
    below_threshold = similarity < threshold
    np.fill_diagonal(below_threshold, False)
    return below_threshold


def truncate_label(label: str, max_length: int) -> str:
    """Truncate a plotting label to a fixed maximum length."""
    if len(label) <= max_length:
        return label
    return label[: max_length - 3] + "..."


def compact_label(label: str) -> str:
    """Preserve a recognised accession and make trailing metadata readable."""
    for pattern in (GENOME_ACCESSION_PATTERN, NUCLEOTIDE_ACCESSION_PATTERN):
        match = pattern.fullmatch(label)
        if match is None:
            continue
        accession, remainder = match.groups()
        if remainder is None:
            return accession
        return f"{accession} | {remainder.replace('_', ' ')}"
    return label.replace("_", " ")


def make_plot_label(label: str, mode: str, max_length: int) -> str:
    """Transform an original MEGA taxon label for plotting."""
    if mode == "original":
        transformed = label
    elif mode == "pretty":
        transformed = label.replace("_", " ")
    elif mode == "compact":
        transformed = compact_label(label)
    else:
        raise ValueError(f"Unknown label mode: {mode}")
    return truncate_label(transformed, max_length)


def calculate_cluster_order(
    p_distance: np.ndarray, method: str
) -> tuple[list[int], np.ndarray]:
    """Calculate one deterministic leaf order from precomputed p-distances."""
    validate_p_distance_matrix(p_distance, p_distance.shape[0])
    if method not in CLUSTER_METHODS:
        raise ValueError(f"Unsupported cluster method: {method}")
    condensed = squareform(p_distance, checks=True)
    linkage_matrix = linkage(
        condensed,
        method=method,
        optimal_ordering=True,
    )
    dendrogram_info = dendrogram(linkage_matrix, no_plot=True)
    return [int(index) for index in dendrogram_info["leaves"]], linkage_matrix


def derive_figure_size(
    taxon_count: int,
    cluster: bool,
    requested_width: float | None,
    requested_height: float | None,
) -> tuple[float, float]:
    """Derive bounded dimensions while respecting explicit overrides."""
    base = min(30.0, max(9.0, 6.0 + taxon_count * 0.28))
    if cluster:
        base += 1.5
    return requested_width or base, requested_height or base


def should_annotate(mode: str, taxon_count: int) -> bool:
    """Resolve the requested cell-annotation mode."""
    if mode == "true":
        return True
    if mode == "false":
        return False
    if mode == "auto":
        return taxon_count <= AUTO_ANNOTATION_MAX_TAXA
    raise ValueError(f"Unknown annotation mode: {mode}")


def _build_figure(
    figure_size: tuple[float, float], cluster: bool
) -> tuple[Figure, Axes, Axes | None, Axes | None, Axes]:
    """Create aligned heatmap, colourbar, and optional dendrogram axes."""
    figure = plt.figure(figsize=figure_size)
    if cluster:
        grid = GridSpec(
            2,
            4,
            figure=figure,
            width_ratios=(1.15, 6.0, 2.4, 0.24),
            height_ratios=(1.15, 6.0),
            left=0.06,
            right=0.94,
            bottom=0.27,
            top=0.90,
            wspace=0.02,
            hspace=0.02,
        )
        top_axis = figure.add_subplot(grid[0, 1])
        left_axis = figure.add_subplot(grid[1, 0])
        heatmap_axis = figure.add_subplot(grid[1, 1])
        colourbar_axis = figure.add_subplot(grid[1, 3])
    else:
        grid = GridSpec(
            1,
            3,
            figure=figure,
            width_ratios=(6.0, 2.4, 0.24),
            left=0.12,
            right=0.94,
            bottom=0.27,
            top=0.90,
            wspace=0.02,
        )
        top_axis = None
        left_axis = None
        heatmap_axis = figure.add_subplot(grid[0, 0])
        colourbar_axis = figure.add_subplot(grid[0, 2])
    return figure, heatmap_axis, top_axis, left_axis, colourbar_axis


def render_heatmap(
    similarity: np.ndarray,
    below_threshold: np.ndarray,
    plot_labels: list[str],
    order: list[int],
    output_path: Path,
    threshold: float,
    dpi: int,
    annotate_mode: str,
    figure_width: float | None,
    figure_height: float | None,
    linkage_matrix: np.ndarray | None,
) -> None:
    """Render a publication-quality similarity heatmap and dendrograms."""
    taxon_count = len(plot_labels)
    cluster = linkage_matrix is not None
    ordered_similarity = similarity[np.ix_(order, order)]
    ordered_below = below_threshold[np.ix_(order, order)]
    ordered_labels = [plot_labels[index] for index in order]
    off_diagonal = ordered_similarity[~np.eye(taxon_count, dtype=bool)]
    vmin = float(math.floor(float(np.min(off_diagonal))))
    if vmin >= 100.0:
        vmin = 99.0

    figure_size = derive_figure_size(
        taxon_count,
        cluster,
        figure_width,
        figure_height,
    )
    figure, heatmap_axis, top_axis, left_axis, colourbar_axis = _build_figure(
        figure_size, cluster
    )
    try:
        if linkage_matrix is not None and top_axis is not None and left_axis is not None:
            dendrogram(
                linkage_matrix,
                ax=top_axis,
                orientation="top",
                no_labels=True,
                color_threshold=0,
                above_threshold_color="black",
                link_color_func=lambda _cluster: "black",
            )
            top_axis.set_xlim(0, taxon_count * 10)
            top_axis.axis("off")
            dendrogram(
                linkage_matrix,
                ax=left_axis,
                orientation="left",
                no_labels=True,
                color_threshold=0,
                above_threshold_color="black",
                link_color_func=lambda _cluster: "black",
            )
            left_axis.set_ylim(taxon_count * 10, 0)
            left_axis.axis("off")

        image = heatmap_axis.imshow(
            ordered_similarity,
            cmap="viridis",
            vmin=vmin,
            vmax=100.0,
            origin="upper",
            interpolation="nearest",
            aspect="equal",
        )
        colourbar = figure.colorbar(image, cax=colourbar_axis)
        colourbar.set_label("16S rRNA gene sequence similarity (%)")

        positions = np.arange(taxon_count)
        label_size = max(4.0, min(8.0, 10.0 - taxon_count * 0.1))
        heatmap_axis.set_xticks(positions, ordered_labels, rotation=90)
        heatmap_axis.set_yticks(positions, ordered_labels)
        heatmap_axis.tick_params(axis="both", labelsize=label_size, length=0)
        heatmap_axis.yaxis.tick_right()
        heatmap_axis.tick_params(axis="y", labelright=True, labelleft=False)
        heatmap_axis.set_xlim(-0.5, taxon_count - 0.5)
        heatmap_axis.set_ylim(taxon_count - 0.5, -0.5)

        for row_index, column_index in np.argwhere(ordered_below):
            heatmap_axis.add_patch(
                Rectangle(
                    (column_index - 0.5, row_index - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor="red",
                    linewidth=0.55,
                )
            )

        if should_annotate(annotate_mode, taxon_count):
            annotation_size = max(2.5, min(7.0, 9.0 - taxon_count * 0.15))
            midpoint = vmin + (100.0 - vmin) * 0.58
            for row_index in range(taxon_count):
                for column_index in range(taxon_count):
                    value = ordered_similarity[row_index, column_index]
                    text_colour = "white" if value <= midpoint else "black"
                    if ordered_below[row_index, column_index]:
                        text_colour = "red"
                    heatmap_axis.text(
                        column_index,
                        row_index,
                        f"{value:.1f}",
                        ha="center",
                        va="center",
                        fontsize=annotation_size,
                        color=text_colour,
                    )

        figure.suptitle(
            "16S rRNA gene sequence similarity from MEGA p-distance",
            fontsize=13,
        )
        figure.text(
            0.5,
            0.015,
            (
                f"Red outline: below {threshold:g}% 16S rRNA gene sequence "
                "similarity threshold"
            ),
            ha="center",
            va="bottom",
            fontsize=9,
            color="red",
        )
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(figure)


def build_output_paths(out_prefix: Path, figure_format: str) -> OutputPaths:
    """Construct all output paths from one prefix."""
    prefix = str(out_prefix)
    return OutputPaths(
        p_distance=Path(prefix + ".pdistance.tsv"),
        similarity=Path(prefix + ".similarity_percent.tsv"),
        below_threshold=Path(prefix + ".below_threshold.tsv"),
        pairwise_long=Path(prefix + ".pairwise_long.tsv"),
        clustered_order=Path(prefix + ".clustered_order.txt"),
        heatmap=Path(prefix + f".heatmap.{figure_format}"),
    )


def write_tabular_outputs(
    labels: list[str],
    plot_labels: list[str],
    p_distance: np.ndarray,
    similarity: np.ndarray,
    below_threshold: np.ndarray,
    order: list[int],
    paths: OutputPaths,
) -> None:
    """Write square matrices, the directed long table, and final taxon order."""
    pd.DataFrame(p_distance, index=labels, columns=labels).to_csv(
        paths.p_distance,
        sep="\t",
        index_label="taxon",
        float_format="%.10f",
        lineterminator="\n",
    )
    pd.DataFrame(similarity, index=labels, columns=labels).to_csv(
        paths.similarity,
        sep="\t",
        index_label="taxon",
        float_format="%.3f",
        lineterminator="\n",
    )
    pd.DataFrame(below_threshold, index=labels, columns=labels).to_csv(
        paths.below_threshold,
        sep="\t",
        index_label="taxon",
        lineterminator="\n",
    )

    records: list[dict[str, str | bool]] = []
    for row_index, taxon1 in enumerate(labels):
        for column_index, taxon2 in enumerate(labels):
            if row_index == column_index:
                continue
            records.append(
                {
                    "taxon1": taxon1,
                    "taxon2": taxon2,
                    "plot_label1": plot_labels[row_index],
                    "plot_label2": plot_labels[column_index],
                    "p_distance": f"{p_distance[row_index, column_index]:.10f}",
                    "similarity_percent": f"{similarity[row_index, column_index]:.3f}",
                    "below_threshold": bool(below_threshold[row_index, column_index]),
                }
            )
    pd.DataFrame.from_records(
        records,
        columns=(
            "taxon1",
            "taxon2",
            "plot_label1",
            "plot_label2",
            "p_distance",
            "similarity_percent",
            "below_threshold",
        ),
    ).to_csv(paths.pairwise_long, sep="\t", index=False, lineterminator="\n")

    ordered_labels = [labels[index] for index in order]
    paths.clustered_order.write_text(
        "\n".join(ordered_labels) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def log_summary(
    labels: list[str],
    similarity: np.ndarray,
    below_threshold: np.ndarray,
    threshold: float,
    paths: OutputPaths,
) -> None:
    """Log extrema, threshold counts, tied pairs, and output locations."""
    taxon_count = len(labels)
    row_indices, column_indices = np.triu_indices(taxon_count, k=1)
    unique_similarities = similarity[row_indices, column_indices]
    minimum = float(np.min(unique_similarities))
    maximum = float(np.max(unique_similarities))
    unique_below = int(np.sum(unique_similarities < threshold))
    directed_below = int(np.sum(below_threshold))

    LOGGER.info("Taxa: %d", taxon_count)
    LOGGER.info("Unique biological pairs: %d", len(unique_similarities))
    LOGGER.info("Directed off-diagonal cells: %d", taxon_count * (taxon_count - 1))
    LOGGER.info("Minimum off-diagonal similarity: %.8f%%", minimum)
    LOGGER.info("Maximum off-diagonal similarity: %.8f%%", maximum)
    LOGGER.info(
        "Below %.8f%%: %d unique pairs; %d directed cells",
        threshold,
        unique_below,
        directed_below,
    )

    for description, target in (("Closest", maximum), ("Most distant", minimum)):
        matches = np.flatnonzero(np.isclose(unique_similarities, target, atol=1e-10))
        for match_index in matches:
            row_index = int(row_indices[match_index])
            column_index = int(column_indices[match_index])
            LOGGER.info(
                "%s pair: %s <> %s = %.8f%%",
                description,
                labels[row_index],
                labels[column_index],
                target,
            )
    for output_path in paths.__dict__.values():
        LOGGER.info("Output: %s", output_path.resolve())


def run(args: argparse.Namespace) -> OutputPaths:
    """Execute parsing, conversion, output writing, plotting, and reporting."""
    data = parse_mega_distance(args.mega_distance)
    similarity = calculate_similarity(data.p_distance)
    below_threshold = calculate_below_threshold(similarity, args.threshold)
    plot_labels = [
        make_plot_label(label, args.label_mode, args.max_label_length)
        for label in data.labels
    ]

    linkage_matrix: np.ndarray | None
    if args.no_cluster:
        order = list(range(len(data.labels)))
        linkage_matrix = None
    else:
        order, linkage_matrix = calculate_cluster_order(
            data.p_distance,
            args.cluster_method,
        )

    paths = build_output_paths(args.out_prefix, args.fig_format)
    try:
        args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
        write_tabular_outputs(
            data.labels,
            plot_labels,
            data.p_distance,
            similarity,
            below_threshold,
            order,
            paths,
        )
        render_heatmap(
            similarity,
            below_threshold,
            plot_labels,
            order,
            paths.heatmap,
            args.threshold,
            args.dpi,
            args.annotate_values,
            args.fig_width,
            args.fig_height,
            linkage_matrix,
        )
    except OSError as error:
        raise ValueError(f"Could not write outputs for prefix {args.out_prefix}: {error}") from error

    log_summary(data.labels, similarity, below_threshold, args.threshold, paths)
    return paths


def main(argv: list[str] | None = None) -> int:
    """Run the command-line application."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args(argv)
    try:
        run(args)
    except ValueError as error:
        LOGGER.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
