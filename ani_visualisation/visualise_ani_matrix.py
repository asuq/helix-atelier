#!/usr/bin/env python3
"""Visualise a FastANI lower-triangular matrix for species separation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NoReturn

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


DEFAULT_LOWER_THRESHOLD = 90.0
DEFAULT_UPPER_THRESHOLD = 100.0
DEFAULT_SPECIES_THRESHOLD = 95.0
DEFAULT_COLOUR_PALETTE = "Blues"
MAX_LABELLED_SAMPLES = 20
MAX_GRIDDED_SAMPLES = 75
MISSING_COLOUR = "#bdbdbd"
OUTPUT_FILENAMES = (
    "ANI_matrix_heatmap.svg",
    "ANI_matrix_heatmap.png",
    "ANI_matrix_heatmap_simple.svg",
    "ANI_matrix_heatmap_simple.png",
)


def die(message: str) -> NoReturn:
    """Exit with a concise error message."""
    raise SystemExit(message)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Visualise a FastANI lower-triangular matrix as clustered and "
            "unclustered species-separation heatmaps."
        )
    )
    parser.add_argument(
        "matrix_path",
        type=Path,
        help="FastANI --matrix lower-triangular PHYLIP matrix.",
    )
    parser.add_argument(
        "--lower-threshold",
        type=float,
        default=DEFAULT_LOWER_THRESHOLD,
        help="Lower heatmap colour limit. Default: 90.",
    )
    parser.add_argument(
        "--upper-threshold",
        type=float,
        default=DEFAULT_UPPER_THRESHOLD,
        help="Upper heatmap colour limit. Default: 100.",
    )
    parser.add_argument(
        "--species-threshold",
        type=float,
        default=DEFAULT_SPECIES_THRESHOLD,
        help="ANI species-reference value marked on the plots. Default: 95.",
    )
    parser.add_argument(
        "--colour-palette",
        "--color-palette",
        dest="colour_palette",
        default=DEFAULT_COLOUR_PALETTE,
        help="Matplotlib colour palette. Default: Blues.",
    )
    parser.add_argument(
        "--linkage",
        choices=["complete", "average"],
        default="complete",
        help=(
            "Hierarchical linkage for ordering. Complete linkage is the strict "
            "default; average linkage is exploratory."
        ),
    )
    args = parser.parse_args(argv)
    validate_thresholds(
        args.lower_threshold,
        args.upper_threshold,
        args.species_threshold,
    )
    validate_colour_palette(args.colour_palette)
    return args


def validate_thresholds(
    lower_threshold: float,
    upper_threshold: float,
    species_threshold: float,
) -> None:
    """Validate the heatmap range and species-reference threshold."""
    values = {
        "Lower threshold": lower_threshold,
        "Upper threshold": upper_threshold,
        "Species threshold": species_threshold,
    }
    for label, value in values.items():
        if not np.isfinite(value) or not 0 <= value <= 100:
            die(f"{label} must be within [0,100], got: {value}")
    if lower_threshold >= upper_threshold:
        die(
            "Lower threshold must be smaller than upper threshold, "
            f"got: {lower_threshold} >= {upper_threshold}"
        )
    if not lower_threshold <= species_threshold <= upper_threshold:
        die(
            "Species threshold must fall within the displayed heatmap range, "
            f"got: {species_threshold} outside "
            f"[{lower_threshold},{upper_threshold}]"
        )


def validate_colour_palette(colour_palette: str) -> None:
    """Validate a Matplotlib colour palette name."""
    if colour_palette not in plt.colormaps():
        die(f"Unknown Matplotlib colour palette: '{colour_palette}'")


def load_phylip_lower_triangular(
    matrix_path: Path,
) -> tuple[list[str], list[list[str]]]:
    """Read and strictly validate a FastANI lower-triangular matrix."""
    if not matrix_path.is_file():
        die(f"ANI matrix file not found: {matrix_path}")

    with matrix_path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    if not lines:
        die("ANI matrix is empty or missing its taxon count.")
    try:
        sample_count = int(lines[0])
    except ValueError:
        die(f"First line must be an integer taxon count; got: '{lines[0]}'")

    if sample_count < 2:
        die("ANI matrix must contain at least two samples.")
    if len(lines) - 1 != sample_count:
        die(f"Expected {sample_count} matrix rows, found {len(lines) - 1}.")

    names: list[str] = []
    rows: list[list[str]] = []
    for row_index in range(sample_count):
        raw_row = lines[row_index + 1]
        expected_values = row_index
        fields = raw_row.rsplit(maxsplit=expected_values)
        if len(fields) != expected_values + 1:
            die(
                f"Row {row_index + 1}: expected {expected_values} values; "
                f"got {len(fields) - 1}. Raw: '{raw_row}'"
            )

        name = fields[0].strip()
        if not name:
            die(f"Row {row_index + 1} has an empty sample name.")
        values = [field.strip() for field in fields[1:]]
        for column_index, raw_value in enumerate(values, start=1):
            if raw_value == "NA":
                continue
            try:
                value = float(raw_value)
            except ValueError:
                die(
                    f"Non-numeric/non-NA token at row '{name}', "
                    f"column {column_index}: '{raw_value}'"
                )
            if not np.isfinite(value) or not 0 <= value <= 100:
                die(
                    f"ANI value out of range [0,100] at row '{name}', "
                    f"column {column_index}: {raw_value}"
                )
        names.append(name)
        rows.append(values)

    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        die("Duplicate name in ANI matrix: " + ", ".join(duplicates))
    return names, rows


def load_matrix(matrix_path: Path) -> tuple[list[str], np.ndarray]:
    """Expand a FastANI lower triangle into a full symmetric matrix."""
    names, rows = load_phylip_lower_triangular(matrix_path)
    matrix_values = np.full((len(names), len(names)), np.nan, dtype=np.float64)
    np.fill_diagonal(matrix_values, 100.0)

    for row_index, values in enumerate(rows):
        for column_index, raw_value in enumerate(values):
            if raw_value == "NA":
                continue
            value = float(raw_value)
            matrix_values[row_index, column_index] = value
            matrix_values[column_index, row_index] = value
    return names, matrix_values


def build_distance_condensed(matrix_values: np.ndarray) -> np.ndarray:
    """Convert ANI percentages to condensed distances, with NA maximally distant."""
    distance_values = np.where(np.isnan(matrix_values), 100.0, 100.0 - matrix_values)
    distance_values[(distance_values < 0) & (distance_values > -1e-8)] = 0.0
    if np.any(distance_values < 0):
        die("Derived negative distances from the ANI matrix.")
    np.fill_diagonal(distance_values, 0.0)
    return squareform(distance_values, checks=False)


def calculate_cluster_order(
    names: list[str],
    matrix_values: np.ndarray,
    linkage_method: str,
) -> tuple[list[int], dict[str, list]]:
    """Return a deterministic dendrogram order and its plotting coordinates."""
    sorted_indices = np.argsort(np.asarray(names), kind="stable")
    sorted_matrix = matrix_values[np.ix_(sorted_indices, sorted_indices)]
    linkage_matrix = linkage(
        build_distance_condensed(sorted_matrix),
        method=linkage_method,
    )
    dendrogram_info = dendrogram(linkage_matrix, no_plot=True)
    ordered_indices = [int(sorted_indices[index]) for index in dendrogram_info["leaves"]]
    return ordered_indices, dendrogram_info


def build_colormap(colour_palette: str) -> colors.Colormap:
    """Build a heatmap palette with a distinct missing-value colour."""
    return plt.get_cmap(colour_palette).with_extremes(bad=MISSING_COLOUR)


def derive_figure_size(sample_count: int, simple: bool = False) -> float:
    """Choose a bounded square figure size for the matrix density."""
    base = 8.0 if simple else 10.0
    maximum = 32.0 if simple else 40.0
    scale = 0.025 if simple else 0.03
    return max(base, min(maximum, base + sample_count * scale))


def derive_label_size(sample_count: int) -> float:
    """Choose a readable label size for labelled matrices."""
    if sample_count <= 10:
        return 7.0
    return 6.0


def derive_clustered_layout(names: list[str]) -> dict[str, float]:
    """Return exact clustered figure and axis geometry in figure fractions."""
    sample_count = len(names)
    base_size = derive_figure_size(sample_count)
    left_outer_in = max(0.12, min(0.2, base_size * 0.012))
    top_outer_in = max(0.1, min(0.16, base_size * 0.01))
    legend_width_in = max(1.0, min(1.4, base_size * 0.09))
    legend_height_in = max(1.2, min(1.7, base_size * 0.105))
    left_dendrogram_width_in = max(0.55, min(0.9, base_size * 0.055))
    top_dendrogram_height_in = max(0.65, min(1.1, base_size * 0.07))
    left_column_width_in = max(legend_width_in, left_dendrogram_width_in)
    top_band_height_in = max(legend_height_in, top_dendrogram_height_in)

    if sample_count <= MAX_LABELLED_SAMPLES:
        label_size = derive_label_size(sample_count)
        longest_label = max(len(name) for name in names)
        label_margin_in = min(
            4.0,
            max(0.55, longest_label * label_size * 0.62 / 72.0 + 0.25),
        )
    else:
        label_margin_in = 0.14

    left_content_in = left_outer_in + left_column_width_in
    top_content_in = top_outer_in + top_band_height_in
    matrix_side_in = min(
        base_size - left_content_in - label_margin_in,
        base_size - top_content_in - label_margin_in,
    )
    if matrix_side_in <= 0:
        raise ValueError("Clustered figure layout leaves no space for the ANI matrix.")

    figure_width_in = left_content_in + matrix_side_in + label_margin_in
    figure_height_in = top_content_in + matrix_side_in + label_margin_in
    matrix_left_in = left_content_in
    matrix_bottom_in = label_margin_in
    matrix_left = matrix_left_in / figure_width_in
    matrix_bottom = matrix_bottom_in / figure_height_in
    matrix_width = matrix_side_in / figure_width_in
    matrix_height = matrix_side_in / figure_height_in

    return {
        "figure_width_in": figure_width_in,
        "figure_height_in": figure_height_in,
        "matrix_left": matrix_left,
        "matrix_bottom": matrix_bottom,
        "matrix_width": matrix_width,
        "matrix_height": matrix_height,
        "top_axis_left": matrix_left,
        "top_axis_bottom": (matrix_bottom_in + matrix_side_in) / figure_height_in,
        "top_axis_width": matrix_width,
        "top_axis_height": top_dendrogram_height_in / figure_height_in,
        "left_axis_left": (
            matrix_left_in - left_dendrogram_width_in
        ) / figure_width_in,
        "left_axis_bottom": matrix_bottom,
        "left_axis_width": left_dendrogram_width_in / figure_width_in,
        "left_axis_height": matrix_height,
        "legend_left": left_outer_in / figure_width_in,
        "legend_bottom": (
            matrix_bottom_in + matrix_side_in + top_band_height_in - legend_height_in
        ) / figure_height_in,
        "legend_width": legend_width_in / figure_width_in,
        "legend_height": legend_height_in / figure_height_in,
    }


def get_heatmap_extent(sample_count: int) -> tuple[float, float, float, float]:
    """Return image bounds that centre samples on integer coordinates."""
    return (-0.5, sample_count - 0.5, sample_count - 0.5, -0.5)


def draw_legend(
    axis: plt.Axes,
    colour_map: colors.Colormap,
    lower_threshold: float,
    upper_threshold: float,
    species_threshold: float,
) -> None:
    """Draw a compact ANI scale with a distinct species-reference marker."""
    gradient = np.linspace(lower_threshold, upper_threshold, 256).reshape(-1, 1)
    axis.imshow(
        gradient,
        cmap=colour_map,
        norm=colors.Normalize(lower_threshold, upper_threshold),
        extent=(0.35, 0.62, lower_threshold, upper_threshold),
        origin="lower",
        aspect="auto",
    )
    axis.add_patch(
        Rectangle(
            (0.35, lower_threshold),
            0.27,
            upper_threshold - lower_threshold,
            fill=False,
            edgecolor="#303030",
            linewidth=0.8,
        )
    )
    axis.plot(
        [0.3, 0.67],
        [species_threshold, species_threshold],
        color="#b2182b",
        linewidth=1.2,
    )
    axis.text(
        0.75,
        species_threshold,
        f"{species_threshold:g}% species\nreference",
        va="center",
        ha="left",
        fontsize=5.5,
        color="#303030",
    )
    legend_ticks = sorted({lower_threshold, species_threshold, upper_threshold})
    axis.set_yticks(legend_ticks)
    axis.set_yticklabels([f"{value:g}" for value in legend_ticks], fontsize=6)
    axis.set_xticks([])
    axis.set_xlim(0, 1.8)
    axis.set_ylim(lower_threshold, upper_threshold)
    axis.tick_params(axis="y", length=0, pad=2)
    axis.text(0.485, upper_threshold, "ANI (%)", ha="center", va="bottom", fontsize=6)
    axis.add_patch(
        Rectangle(
            (0.85, lower_threshold + (upper_threshold - lower_threshold) * 0.12),
            0.14,
            (upper_threshold - lower_threshold) * 0.045,
            facecolor=MISSING_COLOUR,
            edgecolor="#303030",
            linewidth=0.5,
        )
    )
    axis.text(
        1.04,
        lower_threshold + (upper_threshold - lower_threshold) * 0.1425,
        "NA",
        ha="left",
        va="center",
        fontsize=6,
    )
    for spine in axis.spines.values():
        spine.set_visible(False)


def draw_matrix(
    axis: plt.Axes,
    matrix_values: np.ndarray,
    names: list[str],
    colour_map: colors.Colormap,
    normaliser: colors.Normalize,
    show_labels: bool,
    row_labels_right: bool,
) -> None:
    """Draw a square ANI matrix with optional sample labels and grid."""
    axis.imshow(
        np.ma.masked_invalid(matrix_values),
        cmap=colour_map,
        norm=normaliser,
        extent=get_heatmap_extent(len(names)),
        interpolation="nearest",
        origin="upper",
        aspect="equal",
    )
    positions = np.arange(len(names))
    axis.set_xticks(positions)
    axis.set_yticks(positions)
    labels = names if show_labels else [""] * len(names)
    axis.set_xticklabels(labels, rotation=90, fontsize=derive_label_size(len(names)))
    axis.set_yticklabels(labels, fontsize=derive_label_size(len(names)))
    axis.tick_params(length=0, pad=2)
    if row_labels_right:
        axis.yaxis.tick_right()
        for label in axis.get_yticklabels():
            label.set_horizontalalignment("left")
    if len(names) <= MAX_GRIDDED_SAMPLES:
        axis.set_xticks(np.arange(-0.5, len(names), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
        axis.grid(which="minor", color="#ffffff66", linewidth=0.35)
        axis.tick_params(which="minor", bottom=False, left=False)
    for spine in axis.spines.values():
        spine.set_color("#303030")
        spine.set_linewidth(0.8)


def build_dendrogram_segments(
    dendrogram_info: dict[str, list],
    orientation: str,
) -> list[np.ndarray]:
    """Convert SciPy coordinates to segments centred on matrix cells."""
    segments: list[np.ndarray] = []
    for positions, distances in zip(
        dendrogram_info["icoord"],
        dendrogram_info["dcoord"],
        strict=True,
    ):
        remapped = [(position - 5.0) / 10.0 for position in positions]
        if orientation == "top":
            segments.append(np.column_stack([remapped, distances]))
        elif orientation == "left":
            segments.append(np.column_stack([distances, remapped]))
        else:
            raise ValueError(f"Unsupported dendrogram orientation: {orientation}")
    return segments


def draw_dendrogram(
    axis: plt.Axes,
    dendrogram_info: dict[str, list],
    sample_count: int,
    species_threshold: float,
    orientation: str,
) -> None:
    """Draw an aligned dendrogram and mark the species-reference distance."""
    segments = build_dendrogram_segments(dendrogram_info, orientation)
    axis.add_collection(
        LineCollection(
            segments,
            colors="#4a4a4a",
            linewidths=1.0,
            capstyle="butt",
            joinstyle="miter",
        )
    )
    max_distance = max(max(values) for values in dendrogram_info["dcoord"])
    species_distance = 100.0 - species_threshold
    displayed_max = max(max_distance, species_distance * 1.08, 0.1)
    left, right, bottom, top = get_heatmap_extent(sample_count)

    if orientation == "top":
        axis.set_xlim(left, right)
        axis.set_ylim(0, displayed_max)
        axis.axhline(species_distance, color="#b2182b", linewidth=0.9)
        axis.text(
            0.995,
            species_distance,
            f"{species_threshold:g}% ANI",
            ha="right",
            va="bottom",
            fontsize=6,
            color="#303030",
            transform=axis.get_yaxis_transform(),
        )
    elif orientation == "left":
        axis.set_xlim(displayed_max, 0)
        axis.set_ylim(bottom, top)
        axis.axvline(species_distance, color="#b2182b", linewidth=0.9)
        axis.text(
            species_distance,
            0.995,
            f"{species_threshold:g}% ANI",
            ha="left",
            va="top",
            rotation=90,
            fontsize=6,
            color="#303030",
            transform=axis.get_xaxis_transform(),
        )
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.margins(0)


def render_clustered_figure(
    names: list[str],
    matrix_values: np.ndarray,
    output_path: Path,
    lower_threshold: float,
    upper_threshold: float,
    species_threshold: float,
    colour_palette: str,
    linkage_method: str = "complete",
) -> None:
    """Render a clustered heatmap with aligned dendrograms."""
    colour_map = build_colormap(colour_palette)
    normaliser = colors.Normalize(lower_threshold, upper_threshold, clip=True)
    order, dendrogram_info = calculate_cluster_order(names, matrix_values, linkage_method)
    ordered_names = [names[index] for index in order]
    ordered_matrix = matrix_values[np.ix_(order, order)]
    layout = derive_clustered_layout(ordered_names)
    figure = plt.figure(
        figsize=(layout["figure_width_in"], layout["figure_height_in"]),
    )
    legend_axis = figure.add_axes(
        [
            layout["legend_left"],
            layout["legend_bottom"],
            layout["legend_width"],
            layout["legend_height"],
        ]
    )
    matrix_axis = figure.add_axes(
        [
            layout["matrix_left"],
            layout["matrix_bottom"],
            layout["matrix_width"],
            layout["matrix_height"],
        ]
    )
    top_axis = figure.add_axes(
        [
            layout["top_axis_left"],
            layout["top_axis_bottom"],
            layout["top_axis_width"],
            layout["top_axis_height"],
        ]
    )
    left_axis = figure.add_axes(
        [
            layout["left_axis_left"],
            layout["left_axis_bottom"],
            layout["left_axis_width"],
            layout["left_axis_height"],
        ]
    )

    draw_legend(
        legend_axis,
        colour_map,
        lower_threshold,
        upper_threshold,
        species_threshold,
    )
    draw_matrix(
        matrix_axis,
        ordered_matrix,
        ordered_names,
        colour_map,
        normaliser,
        show_labels=len(names) <= MAX_LABELLED_SAMPLES,
        row_labels_right=True,
    )
    draw_dendrogram(
        top_axis,
        dendrogram_info,
        len(names),
        species_threshold,
        "top",
    )
    draw_dendrogram(
        left_axis,
        dendrogram_info,
        len(names),
        species_threshold,
        "left",
    )
    figure.savefig(output_path, dpi=300, facecolor="white")
    plt.close(figure)


def render_simple_figure(
    names: list[str],
    matrix_values: np.ndarray,
    output_path: Path,
    lower_threshold: float,
    upper_threshold: float,
    species_threshold: float,
    colour_palette: str,
) -> None:
    """Render the matrix in its original order without dendrograms or labels."""
    colour_map = build_colormap(colour_palette)
    normaliser = colors.Normalize(lower_threshold, upper_threshold, clip=True)
    figure_size = derive_figure_size(len(names), simple=True)
    figure = plt.figure(figsize=(figure_size, figure_size))
    grid = GridSpec(
        1,
        2,
        width_ratios=[1.8, 12],
        wspace=0.04,
        left=0.04,
        right=0.99,
        bottom=0.04,
        top=0.99,
    )
    legend_axis = figure.add_subplot(grid[0, 0])
    matrix_axis = figure.add_subplot(grid[0, 1])
    draw_legend(
        legend_axis,
        colour_map,
        lower_threshold,
        upper_threshold,
        species_threshold,
    )
    draw_matrix(
        matrix_axis,
        matrix_values,
        names,
        colour_map,
        normaliser,
        show_labels=False,
        row_labels_right=False,
    )
    figure.savefig(output_path, dpi=300, facecolor="white")
    plt.close(figure)


def write_outputs(
    names: list[str],
    matrix_values: np.ndarray,
    matrix_path: Path,
    lower_threshold: float,
    upper_threshold: float,
    species_threshold: float,
    colour_palette: str,
    linkage_method: str,
) -> None:
    """Write clustered and simple SVG/PNG figures beside the input matrix."""
    output_paths = [matrix_path.parent / filename for filename in OUTPUT_FILENAMES]
    for output_path in output_paths[:2]:
        render_clustered_figure(
            names,
            matrix_values,
            output_path,
            lower_threshold,
            upper_threshold,
            species_threshold,
            colour_palette,
            linkage_method,
        )
    for output_path in output_paths[2:]:
        render_simple_figure(
            names,
            matrix_values,
            output_path,
            lower_threshold,
            upper_threshold,
            species_threshold,
            colour_palette,
        )
    for output_path in output_paths:
        print(f"Wrote {output_path}")


def main(argv: list[str] | None = None) -> int:
    """Run the ANI species-separation visualisation workflow."""
    matplotlib.rcParams["svg.fonttype"] = "none"
    args = parse_args(argv)
    names, matrix_values = load_matrix(args.matrix_path)
    write_outputs(
        names,
        matrix_values,
        args.matrix_path,
        args.lower_threshold,
        args.upper_threshold,
        args.species_threshold,
        args.colour_palette,
        args.linkage,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
