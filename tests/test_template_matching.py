from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.ad_detector.matching import combine_detections
from scripts.ad_detector.boundary.cuts import dark_transition_bounds
from scripts.ad_detector.templates import load_templates, template_match_edge_debug


class TemplateMatchingTests(unittest.TestCase):
    def test_overlapping_template_candidates_keep_best_window(self) -> None:
        detections = [
            {
                "start": 288.0,
                "end": 313.7,
                "score": 0.98,
                "sources": ["template=4:56-5:22"],
                "kind": "template_library",
            },
            {
                "start": 296.0,
                "end": 321.7,
                "score": 0.999,
                "sources": ["template=4:56-5:22"],
                "kind": "template_library",
            },
        ]

        combined = combine_detections(detections, merge_gap=8.0)

        self.assertEqual(len(combined), 1)
        self.assertEqual(combined[0]["start"], 296.0)
        self.assertEqual(combined[0]["end"], 321.7)
        self.assertEqual(combined[0]["score"], 0.999)

    def test_duplicate_template_files_are_loaded_once(self) -> None:
        features = np.eye(4, dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("one.npz", "two.npz"):
                np.savez_compressed(
                    root / name,
                    features=features,
                    duration=np.asarray([4.0], dtype=np.float32),
                    sample_rate=np.asarray([1.0], dtype=np.float32),
                    source=np.asarray(["1:00-1:04"]),
                )

            self.assertEqual(len(load_templates(root)), 1)

    def test_edge_debug_reports_inside_outside_contrast(self) -> None:
        template = np.asarray([[1.0, 0.0]] * 4, dtype=np.float32)
        content = np.asarray([[0.0, 1.0]] * 3, dtype=np.float32)
        features = np.vstack([content, template, content])

        debug = template_match_edge_debug(features, template, score_index=3, edge_samples=3)

        self.assertEqual(debug["template_start_inside_similarity"], 1.0)
        self.assertEqual(debug["template_start_outside_similarity"], 0.0)
        self.assertEqual(debug["template_start_contrast"], 1.0)
        self.assertEqual(debug["template_end_contrast"], 1.0)

    def test_dark_transition_bounds_exclude_black_frames_from_end(self) -> None:
        rows = [
            (10.000, 0.5, 0.0, 0.1),
            (10.042, 0.0, 1.0, 0.8),
            (10.083, 0.0, 1.0, 0.0),
            (10.125, 0.6, 0.0, 0.8),
        ]

        bounds = dark_transition_bounds(rows, center=10.083, sample_rate=24.0)

        self.assertIsNotNone(bounds)
        assert bounds is not None
        self.assertAlmostEqual(bounds[0], 10.000333, places=3)
        self.assertAlmostEqual(bounds[1], 10.124667, places=3)


if __name__ == "__main__":
    unittest.main()
