"""Tests for the MEGA 16S similarity heatmap tool."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "make_16s_similarity_heatmap.py"


def load_module():
    """Load the standalone script for focused helper tests."""
    spec = importlib.util.spec_from_file_location("make_16s_similarity_heatmap", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_module()


def write_mega(path: Path, labels: list[str], rows: list[str]) -> None:
    """Write a minimal ASCII MEGA lower-left matrix fixture."""
    label_lines = "\n".join(
        f"[{index:2d}] #{label}" for index, label in enumerate(labels, start=1)
    )
    matrix_lines = "\n".join(
        f"[{index:2d}] {row}" for index, row in enumerate(rows, start=1)
    )
    path.write_text(
        "\n".join(
            (
                "#mega",
                "!Title: fixture;",
                f"!Format DataType=Distance DataFormat=LowerLeft NTaxa={len(labels)};",
                label_lines,
                "[             1          2          3 ]",
                matrix_lines,
                "",
            )
        ),
        encoding="ascii",
    )


@pytest.fixture
def three_taxon_mega(tmp_path: Path) -> Path:
    """Return a representative three-taxon MEGA matrix."""
    path = tmp_path / "distance.meg"
    write_mega(
        path,
        [
            "GCA_900230115_1",
            "NR_041373.1_Emticicia_ginsengisoli_strain_Gsoil_085",
            "ERR13863366_binette_bin9",
        ],
        ["", "0.0135", "0.0200 0.0000"],
    )
    return path


def test_parse_args_uses_requested_defaults() -> None:
    args = TOOL.parse_args(
        ["--mega-distance", "input.meg", "--out-prefix", "output"]
    )

    assert args.threshold == 98.65
    assert args.fig_format == "pdf"
    assert args.dpi == 300
    assert args.cluster_method == "average"
    assert args.label_mode == "compact"
    assert args.max_label_length == 90
    assert args.annotate_values == "auto"
    assert not args.no_cluster


@pytest.mark.parametrize(
    ("extra_args", "message"),
    [
        (["--threshold", "nan"], "within [0, 100]"),
        (["--dpi", "0"], "greater than zero"),
        (["--max-label-length", "3"], "at least 4"),
        (["--fig-width", "-1"], "finite and positive"),
    ],
)
def test_parse_args_rejects_invalid_numeric_options(
    extra_args: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        TOOL.parse_args(
            ["--mega-distance", "input.meg", "--out-prefix", "output", *extra_args]
        )
    assert message in capsys.readouterr().err


def test_parser_extracts_labels_and_expands_lower_triangle(
    three_taxon_mega: Path,
) -> None:
    data = TOOL.parse_mega_distance(three_taxon_mega)

    assert data.labels == [
        "GCA_900230115_1",
        "NR_041373.1_Emticicia_ginsengisoli_strain_Gsoil_085",
        "ERR13863366_binette_bin9",
    ]
    np.testing.assert_allclose(
        data.p_distance,
        np.array(
            [
                [0.0, 0.0135, 0.0200],
                [0.0135, 0.0, 0.0],
                [0.0200, 0.0, 0.0],
            ]
        ),
    )


def test_parser_rejects_missing_ntaxa(tmp_path: Path) -> None:
    path = tmp_path / "missing.meg"
    path.write_text("#mega\n[ 1] #A\n[ 1]\n", encoding="ascii")
    with pytest.raises(ValueError, match="No !Format"):
        TOOL.parse_mega_distance(path)


def test_parser_rejects_format_without_ntaxa(tmp_path: Path) -> None:
    path = tmp_path / "missing.meg"
    path.write_text("#mega\n!Format DataType=Distance;\n", encoding="ascii")
    with pytest.raises(ValueError, match="lacks NTaxa"):
        TOOL.parse_mega_distance(path)


@pytest.mark.parametrize(
    ("labels", "rows", "message"),
    [
        (["A", "A"], ["", "0.1"], "Duplicate taxon label"),
        (["A", "B", "C"], ["", "0.1", "0.2"], "expected 2 value"),
        (["A", "B"], ["", "unknown"], "non-numeric p-distance"),
        (["A", "B"], ["", "1.1"], r"outside \[0, 1\]"),
        (["A"], [""], "at least 2"),
    ],
)
def test_parser_rejects_invalid_matrices(
    tmp_path: Path, labels: list[str], rows: list[str], message: str
) -> None:
    path = tmp_path / "invalid.meg"
    write_mega(path, labels, rows)
    with pytest.raises(ValueError, match=message):
        TOOL.parse_mega_distance(path)


def test_parser_rejects_duplicate_label_index(tmp_path: Path) -> None:
    path = tmp_path / "duplicate-index.meg"
    path.write_text(
        "\n".join(
            (
                "#mega",
                "!Format DataType=Distance DataFormat=LowerLeft NTaxa=2;",
                "[ 1] #A",
                "[ 1] #B",
                "[ 2] #C",
                "[ 1]",
                "[ 2] 0.1",
                "",
            )
        ),
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="duplicate taxon-label index"):
        TOOL.parse_mega_distance(path)


def test_similarity_and_threshold_use_unrounded_values() -> None:
    p_distance = np.array([[0.0, 0.0135], [0.0135, 0.0]])
    similarity = TOOL.calculate_similarity(p_distance)

    np.testing.assert_allclose(similarity, [[100.0, 98.65], [98.65, 100.0]])
    assert not TOOL.calculate_below_threshold(similarity, 98.65).any()
    assert TOOL.calculate_below_threshold(similarity, 98.6500001)[0, 1]


@pytest.mark.parametrize(
    ("label", "mode", "expected"),
    [
        (
            "NR_041373.1_Emticicia_ginsengisoli_strain_Gsoil_085",
            "original",
            "NR_041373.1_Emticicia_ginsengisoli_strain_Gsoil_085",
        ),
        (
            "NR_041373.1_Emticicia_ginsengisoli_strain_Gsoil_085",
            "pretty",
            "NR 041373.1 Emticicia ginsengisoli strain Gsoil 085",
        ),
        (
            "NR_041373.1_Emticicia_ginsengisoli_strain_Gsoil_085",
            "compact",
            "NR_041373.1 | Emticicia ginsengisoli strain Gsoil 085",
        ),
        ("GCA_900230115_1", "compact", "GCA_900230115_1"),
        ("ERR13863366_binette_bin9", "compact", "ERR13863366 binette bin9"),
    ],
)
def test_label_modes(label: str, mode: str, expected: str) -> None:
    assert TOOL.make_plot_label(label, mode, 90) == expected


def test_label_truncation_is_deterministic() -> None:
    assert TOOL.make_plot_label("ABCDEFGHIJK", "original", 8) == "ABCDE..."


def test_cluster_order_is_deterministic_and_complete() -> None:
    matrix = np.array(
        [
            [0.0, 0.01, 0.20],
            [0.01, 0.0, 0.19],
            [0.20, 0.19, 0.0],
        ]
    )
    first, first_linkage = TOOL.calculate_cluster_order(matrix, "average")
    second, second_linkage = TOOL.calculate_cluster_order(matrix, "average")

    assert first == second
    assert sorted(first) == [0, 1, 2]
    np.testing.assert_allclose(first_linkage, second_linkage)


@pytest.mark.parametrize(
    ("mode", "taxon_count", "expected"),
    [("true", 100, True), ("false", 2, False), ("auto", 40, True), ("auto", 41, False)],
)
def test_annotation_modes(mode: str, taxon_count: int, expected: bool) -> None:
    assert TOOL.should_annotate(mode, taxon_count) is expected


@pytest.mark.parametrize("cluster", [True, False])
def test_layout_reserves_space_for_right_side_taxon_labels(cluster: bool) -> None:
    figure, heatmap_axis, _, _, colourbar_axis = TOOL._build_figure((12.0, 12.0), cluster)
    try:
        label_margin = colourbar_axis.get_position().x0 - heatmap_axis.get_position().x1
        assert label_margin > 0.20
    finally:
        plt.close(figure)


def test_no_cluster_run_preserves_original_order(
    three_taxon_mega: Path, tmp_path: Path
) -> None:
    out_prefix = tmp_path / "nested" / "result"
    args = TOOL.parse_args(
        [
            "--mega-distance",
            str(three_taxon_mega),
            "--out-prefix",
            str(out_prefix),
            "--no-cluster",
            "--fig-format",
            "svg",
            "--annotate-values",
            "false",
        ]
    )
    paths = TOOL.run(args)

    assert paths.clustered_order.read_text(encoding="utf-8").splitlines() == [
        "GCA_900230115_1",
        "NR_041373.1_Emticicia_ginsengisoli_strain_Gsoil_085",
        "ERR13863366_binette_bin9",
    ]


def test_outputs_have_requested_schemas_and_directed_rows(
    three_taxon_mega: Path, tmp_path: Path
) -> None:
    out_prefix = tmp_path / "result"
    args = TOOL.parse_args(
        [
            "--mega-distance",
            str(three_taxon_mega),
            "--out-prefix",
            str(out_prefix),
            "--fig-format",
            "svg",
            "--annotate-values",
            "false",
        ]
    )
    paths = TOOL.run(args)

    for path in paths.__dict__.values():
        assert path.is_file()
        assert path.stat().st_size > 0
    similarity_text = paths.similarity.read_text(encoding="utf-8")
    assert "98.650" in similarity_text
    below = pd.read_csv(paths.below_threshold, sep="\t", index_col=0)
    assert not np.diag(below.to_numpy(dtype=bool)).any()
    pairwise = pd.read_csv(paths.pairwise_long, sep="\t")
    assert pairwise.columns.tolist() == [
        "taxon1",
        "taxon2",
        "plot_label1",
        "plot_label2",
        "p_distance",
        "similarity_percent",
        "below_threshold",
    ]
    assert len(pairwise) == 6
    assert not (pairwise["taxon1"] == pairwise["taxon2"]).any()
    svg = paths.heatmap.read_text(encoding="utf-8")
    assert "Black grid; values below 98.65%" in svg
    assert "16S rRNA gene sequence similarity (%)" in svg


def test_render_draws_an_always_black_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    similarity = np.array([[100.0, 97.0], [97.0, 100.0]])
    below = TOOL.calculate_below_threshold(similarity, 98.65)
    recorded_rectangles: list[object] = []
    original_add_patch = plt.Axes.add_patch

    def record_patch(axis, patch):
        recorded_rectangles.append(patch)
        return original_add_patch(axis, patch)

    monkeypatch.setattr(plt.Axes, "add_patch", record_patch)
    TOOL.render_heatmap(
        similarity,
        below,
        ["A", "B"],
        [0, 1],
        tmp_path / "heatmap.svg",
        98.65,
        300,
        "false",
        None,
        None,
        None,
    )

    cell_rectangles = [
        patch
        for patch in recorded_rectangles
        if isinstance(patch, TOOL.Rectangle)
    ]
    assert len(cell_rectangles) == 4
    assert all(
        patch.get_edgecolor() == (0.0, 0.0, 0.0, 1.0)
        for patch in cell_rectangles
    )


def test_cli_reports_missing_input(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--mega-distance",
            str(tmp_path / "missing.meg"),
            "--out-prefix",
            str(tmp_path / "result"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "MEGA distance file not found" in result.stderr
