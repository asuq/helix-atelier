"""Tests for the FastANI species-separation visualiser."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_PATH = Path(__file__).with_name("visualise_ani_matrix.py")
EXPECTED_OUTPUTS = (
    "ANI_matrix_heatmap.svg",
    "ANI_matrix_heatmap.png",
    "ANI_matrix_heatmap_simple.svg",
    "ANI_matrix_heatmap_simple.png",
)


def load_visualiser_module():
    """Load the visualiser as a module for focused helper tests."""
    spec = importlib.util.spec_from_file_location("visualise_ani_matrix", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VISUALISER = load_visualiser_module()


def write_matrix(path: Path, rows: list[str]) -> None:
    """Write an ASCII FastANI matrix fixture."""
    path.write_text("\n".join(rows) + "\n", encoding="ascii")


class VisualiseANIMatrixTests(unittest.TestCase):
    """Verify ANI parsing, clustering, validation, and rendering."""

    def test_parse_args_uses_species_separation_defaults(self) -> None:
        args = VISUALISER.parse_args(["matrix.txt"])

        self.assertIsNone(args.lower_threshold)
        self.assertEqual(args.lower_threshold_mode, "static")
        self.assertEqual(args.upper_threshold, 100.0)
        self.assertEqual(args.species_threshold, 95.0)
        self.assertEqual(args.colour_palette, "Blues")
        self.assertEqual(args.linkage, "average")

    def test_static_mode_resolves_default_and_explicit_thresholds(self) -> None:
        matrix = np.array([[100.0, 92.5], [92.5, 100.0]])

        self.assertEqual(
            VISUALISER.resolve_lower_threshold(matrix, "static", None),
            75.0,
        )
        self.assertEqual(
            VISUALISER.resolve_lower_threshold(matrix, "static", 90.0),
            90.0,
        )

    def test_dynamic_mode_floors_lowest_finite_pairwise_ani(self) -> None:
        matrix = np.array(
            [
                [100.0, 75.859894, np.nan],
                [75.859894, 100.0, 93.2],
                [np.nan, 93.2, 100.0],
            ]
        )

        threshold = VISUALISER.resolve_lower_threshold(matrix, "dynamic", None)

        self.assertEqual(threshold, 75.0)

    def test_dynamic_mode_rejects_matrix_without_finite_pairs(self) -> None:
        matrix = np.array([[100.0, np.nan], [np.nan, 100.0]])

        with self.assertRaisesRegex(SystemExit, "no finite off-diagonal ANI"):
            VISUALISER.resolve_lower_threshold(matrix, "dynamic", None)

    def test_dynamic_mode_rejects_explicit_lower_threshold(self) -> None:
        with self.assertRaisesRegex(SystemExit, "cannot be combined"):
            VISUALISER.parse_args(
                [
                    "--lower-threshold-mode",
                    "dynamic",
                    "--lower-threshold",
                    "80",
                    "matrix.txt",
                ]
            )

    def test_parser_preserves_names_with_spaces_and_expands_na(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            matrix_path = Path(tempdir) / "fastani.matrix"
            write_matrix(
                matrix_path,
                [
                    "3",
                    "Genome alpha",
                    "Genome beta 96.2",
                    "Genome gamma NA 94.5",
                ],
            )

            names, matrix = VISUALISER.load_matrix(matrix_path)

        self.assertEqual(names, ["Genome alpha", "Genome beta", "Genome gamma"])
        self.assertEqual(matrix[0, 1], 96.2)
        self.assertTrue(np.isnan(matrix[0, 2]))
        self.assertEqual(matrix[1, 2], 94.5)
        self.assertTrue(np.all(np.diag(matrix) == 100.0))

    def test_missing_ani_is_maximum_clustering_distance(self) -> None:
        matrix = np.array([[100.0, np.nan], [np.nan, 100.0]])

        condensed = VISUALISER.build_distance_condensed(matrix)

        np.testing.assert_array_equal(condensed, np.array([100.0]))

    def test_missing_ani_uses_neutral_colour(self) -> None:
        colour_map = VISUALISER.build_colormap("Blues")

        np.testing.assert_allclose(
            colour_map.get_bad(),
            plt.matplotlib.colors.to_rgba("#bdbdbd"),
        )

    def test_threshold_validation_rejects_species_reference_outside_display(self) -> None:
        with self.assertRaisesRegex(SystemExit, "must fall within"):
            VISUALISER.validate_thresholds(96.0, 100.0, 95.0)

    def test_threshold_validation_rejects_inverted_range(self) -> None:
        with self.assertRaisesRegex(SystemExit, "must be smaller"):
            VISUALISER.validate_thresholds(100.0, 90.0, 95.0)

    def test_unknown_palette_is_rejected(self) -> None:
        with self.assertRaisesRegex(SystemExit, "Unknown Matplotlib"):
            VISUALISER.validate_colour_palette("not_a_palette")

    def test_duplicate_samples_are_rejected(self) -> None:
        self.assert_matrix_failure(
            ["2", "Genome A", "Genome A 96.0"],
            "Duplicate name",
        )

    def test_malformed_triangle_is_rejected(self) -> None:
        self.assert_matrix_failure(
            ["3", "A", "B 96.0", "C 94.0"],
            "expected 2 values",
        )

    def test_non_numeric_value_is_rejected(self) -> None:
        self.assert_matrix_failure(
            ["2", "A", "B unknown"],
            "Non-numeric/non-NA",
        )

    def test_out_of_range_value_is_rejected(self) -> None:
        self.assert_matrix_failure(
            ["2", "A", "B 101"],
            "out of range",
        )

    def test_cluster_order_is_deterministic_for_unsorted_names(self) -> None:
        names = ["C", "A", "B"]
        matrix = np.array(
            [
                [100.0, 92.0, 91.0],
                [92.0, 100.0, 98.0],
                [91.0, 98.0, 100.0],
            ]
        )

        first_order, _ = VISUALISER.calculate_cluster_order(names, matrix, "complete")
        second_order, _ = VISUALISER.calculate_cluster_order(names, matrix, "complete")

        self.assertEqual(first_order, second_order)

    def test_clustered_layout_places_dendrograms_flush_with_matrix(self) -> None:
        layout = VISUALISER.derive_clustered_layout(["A", "B", "C"])

        self.assertAlmostEqual(
            layout["top_axis_bottom"],
            layout["matrix_bottom"] + layout["matrix_height"],
        )
        self.assertAlmostEqual(layout["top_axis_left"], layout["matrix_left"])
        self.assertAlmostEqual(layout["top_axis_width"], layout["matrix_width"])
        self.assertAlmostEqual(
            layout["left_axis_left"] + layout["left_axis_width"],
            layout["matrix_left"],
        )
        self.assertAlmostEqual(layout["left_axis_bottom"], layout["matrix_bottom"])
        self.assertAlmostEqual(layout["left_axis_height"], layout["matrix_height"])

    def test_large_na_matrix_renders_with_aligned_unlabelled_layout(self) -> None:
        sample_count = 24
        names = [f"sample_{index:02d}" for index in range(sample_count)]
        matrix = np.full((sample_count, sample_count), 92.0)
        np.fill_diagonal(matrix, 100.0)
        matrix[:12, :12] = 97.0
        matrix[12:, 12:] = 96.0
        np.fill_diagonal(matrix, 100.0)
        matrix[0, 23] = np.nan
        matrix[23, 0] = np.nan

        with tempfile.TemporaryDirectory() as tempdir:
            output_path = Path(tempdir) / "large.svg"
            VISUALISER.render_clustered_figure(
                names,
                matrix,
                output_path,
                90.0,
                100.0,
                95.0,
                "Blues",
            )
            svg_text = output_path.read_text(encoding="utf-8")

        self.assertNotIn("sample_00", svg_text)
        self.assertIn("95% ANI", svg_text)

    def test_clustered_render_defaults_to_average_linkage(self) -> None:
        names = ["A", "B", "C"]
        matrix = np.array(
            [
                [100.0, 98.0, 91.0],
                [98.0, 100.0, 92.0],
                [91.0, 92.0, 100.0],
            ]
        )
        with tempfile.TemporaryDirectory() as tempdir, mock.patch.object(
            VISUALISER,
            "linkage",
            wraps=VISUALISER.linkage,
        ) as linkage_mock:
            VISUALISER.render_clustered_figure(
                names,
                matrix,
                Path(tempdir) / "clustered.svg",
                90.0,
                100.0,
                95.0,
                "Blues",
            )

        self.assertEqual(linkage_mock.call_args.kwargs["method"], "average")

    def test_clustered_render_accepts_complete_linkage(self) -> None:
        names = ["A", "B", "C"]
        matrix = np.array(
            [
                [100.0, 98.0, 91.0],
                [98.0, 100.0, 92.0],
                [91.0, 92.0, 100.0],
            ]
        )
        with tempfile.TemporaryDirectory() as tempdir, mock.patch.object(
            VISUALISER,
            "linkage",
            wraps=VISUALISER.linkage,
        ) as linkage_mock:
            VISUALISER.render_clustered_figure(
                names,
                matrix,
                Path(tempdir) / "clustered.svg",
                90.0,
                100.0,
                95.0,
                "Blues",
                "complete",
            )

        self.assertEqual(linkage_mock.call_args.kwargs["method"], "complete")

    def test_cli_writes_all_outputs_and_marks_species_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            matrix_path = temp_path / "fastani.matrix"
            write_matrix(
                matrix_path,
                [
                    "3",
                    "Genome A",
                    "Genome B 97.5",
                    "Genome C NA 93.0",
                ],
            )

            result = subprocess.run(
                [sys.executable, str(SCRIPT_PATH), str(matrix_path)],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            for filename in EXPECTED_OUTPUTS:
                output_path = temp_path / filename
                self.assertTrue(output_path.is_file(), msg=f"Missing {output_path}")
                self.assertGreater(output_path.stat().st_size, 0)
            clustered_svg = (temp_path / EXPECTED_OUTPUTS[0]).read_text(encoding="utf-8")
            self.assertIn(">75<", clustered_svg)
            self.assertIn(">95<", clustered_svg)
            self.assertIn(">100<", clustered_svg)
            self.assertIn("95% species", clustered_svg)
            self.assertIn("95% ANI", clustered_svg)
            self.assertIn("Genome A", clustered_svg)
            self.assertIn("75% (static mode)", result.stdout)

    def test_cli_dynamic_mode_uses_matrix_minimum(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            temp_path = Path(tempdir)
            matrix_path = temp_path / "fastani.matrix"
            write_matrix(
                matrix_path,
                [
                    "3",
                    "Genome A",
                    "Genome B 97.5",
                    "Genome C NA 93.2",
                ],
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "--lower-threshold-mode",
                    "dynamic",
                    str(matrix_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            clustered_svg = (temp_path / EXPECTED_OUTPUTS[0]).read_text(encoding="utf-8")
            self.assertIn(">93<", clustered_svg)
            self.assertIn("93% (dynamic mode)", result.stdout)

    def assert_matrix_failure(self, rows: list[str], expected_error: str) -> None:
        """Assert that malformed matrix input fails with an actionable message."""
        with tempfile.TemporaryDirectory() as tempdir:
            matrix_path = Path(tempdir) / "fastani.matrix"
            write_matrix(matrix_path, rows)
            with self.assertRaisesRegex(SystemExit, expected_error):
                VISUALISER.load_matrix(matrix_path)


if __name__ == "__main__":
    unittest.main()
