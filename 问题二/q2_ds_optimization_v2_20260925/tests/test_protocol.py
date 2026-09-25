#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from q2hc_select import (  # noqa: E402
    confirm_ensemble,
    fold_indices,
    read_json,
    select_ensemble,
    selector_indices,
    summarize,
)


class SplitTests(unittest.TestCase):
    def test_grouped_partitions_do_not_leak(self):
        ids = [f"video{i // 3}$_${i % 3}" for i in range(90)]
        train, holdout = fold_indices(
            ids, folds=3, fold=1, seed=20260925, separator="$_$"
        )
        train_groups = {ids[i].split("$_$")[0] for i in train}
        holdout_groups = {ids[i].split("$_$")[0] for i in holdout}
        self.assertFalse(train_groups & holdout_groups)
        selector, confirmer = selector_indices(ids, 20260925, 0.5, "$_$")
        selector_groups = {ids[i].split("$_$")[0] for i in selector}
        confirmer_groups = {ids[i].split("$_$")[0] for i in confirmer}
        self.assertFalse(selector_groups & confirmer_groups)


class MetricTests(unittest.TestCase):
    def test_worst_f1_is_minimum(self):
        labels = np.asarray([0, 1, 2, 0, 1, 2])
        regression = np.asarray([-1, 0, 1, -1, 0, 1], dtype=np.float32)
        condition_ids = ["clean", "text|0.1|r0", "text|0.3|r0"]
        probabilities = np.zeros((3, 6, 3), dtype=np.float32)
        probabilities[0, np.arange(6), labels] = 1.0
        probabilities[1] = probabilities[0]
        probabilities[2, :, 1] = 1.0
        raw = np.tile(regression, (3, 1))
        summary = summarize(
            probabilities,
            raw,
            condition_ids,
            labels,
            regression,
            np.arange(6),
        )
        cell_f1 = [row["macro_f1"] for row in summary["cells"]]
        self.assertEqual(summary["worst_condition_F1"], min(cell_f1))


class SelectionTests(unittest.TestCase):
    def _write_cache(self, path, member_id, ids, condition_ids, probabilities, raw):
        np.savez_compressed(
            path,
            member_id=np.asarray(member_id),
            checkpoint=np.asarray(f"/tmp/{member_id}.pt"),
            checkpoint_sha256=np.asarray(member_id.zfill(64)[:64]),
            ids=np.asarray(ids),
            condition_ids=np.asarray(condition_ids),
            probabilities=probabilities,
            raw=raw,
        )

    def test_candidate_selection_and_confirmation(self):
        config = read_json(ROOT / "configs" / "protocol.json")
        config["selection"]["min_slots"] = 5
        config["selection"]["max_slots"] = 13
        config["selection"]["max_pool_after_individual_screen"] = 20
        n = 120
        ids = [f"video{i}$_$0" for i in range(n)]
        labels = np.asarray([i % 3 for i in range(n)], dtype=np.int64)
        regression = np.asarray(
            [(-1.0, 0.0, 1.0)[i % 3] for i in range(n)], dtype=np.float32
        )
        condition_ids = ["clean", "text|0.1|r0", "text|0.3|r0"]
        baseline_probabilities = np.full((3, n, 3), 0.3, dtype=np.float32)
        baseline_probabilities[:, :, 1] = 0.4
        baseline_raw = np.zeros((3, n), dtype=np.float32)
        candidate_probabilities = np.full((3, n, 3), 0.01, dtype=np.float32)
        for condition_index in range(3):
            candidate_probabilities[condition_index, np.arange(n), labels] = 0.98
        candidate_raw = np.tile(regression, (3, 1))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data.npz"
            np.savez_compressed(
                data,
                valid_ids=np.asarray(ids),
                valid_labels=labels,
                valid_regression=regression,
            )
            select_paths = []
            confirm_paths = []
            for item in config["baseline"]["checkpoints"]:
                for bank, collection in (
                    ("select", select_paths),
                    ("confirm", confirm_paths),
                ):
                    path = root / f"{bank}_{item['id']}.npz"
                    self._write_cache(
                        path,
                        item["id"],
                        ids,
                        condition_ids,
                        baseline_probabilities,
                        baseline_raw,
                    )
                    collection.append(path)
            for bank, collection in (
                ("select", select_paths),
                ("confirm", confirm_paths),
            ):
                path = root / f"{bank}_new_seed101.npz"
                self._write_cache(
                    path,
                    "new_seed101",
                    ids,
                    condition_ids,
                    candidate_probabilities,
                    candidate_raw,
                )
                collection.append(path)
            selection_path = root / "selection.json"
            selection = select_ensemble(config, select_paths, data, selection_path)
            self.assertIn("new_seed101", selection["selected"]["weights"])
            confirmation = confirm_ensemble(
                config,
                confirm_paths,
                data,
                selection_path,
                root / "confirmation.json",
            )
            self.assertTrue(confirmation["passed"])
            self.assertEqual(confirmation["frozen_choice"], "candidate")

    def test_confirmation_rejects_non_improving_candidate(self):
        config = read_json(ROOT / "configs" / "protocol.json")
        n = 90
        ids = [f"group{i}$_$0" for i in range(n)]
        labels = np.asarray([i % 3 for i in range(n)], dtype=np.int64)
        regression = np.asarray(
            [(-1.0, 0.0, 1.0)[i % 3] for i in range(n)], dtype=np.float32
        )
        condition_ids = ["clean", "text|0.1|r0"]
        probabilities = np.full((2, n, 3), 0.01, dtype=np.float32)
        for condition_index in range(2):
            probabilities[condition_index, np.arange(n), labels] = 0.98
        raw = np.tile(regression, (2, 1))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data.npz"
            np.savez_compressed(
                data,
                valid_ids=np.asarray(ids),
                valid_labels=labels,
                valid_regression=regression,
            )
            paths = []
            weights = {}
            members = {}
            for item in config["baseline"]["checkpoints"]:
                path = root / f"{item['id']}.npz"
                self._write_cache(
                    path, item["id"], ids, condition_ids, probabilities, raw
                )
                paths.append(path)
                weights[item["id"]] = 1.0 / 9.0
                members[item["id"]] = {
                    "checkpoint": f"/tmp/{item['id']}.pt",
                    "checkpoint_sha256": item["id"].zfill(64)[:64],
                    "cache": str(path),
                }
            selection_path = root / "selection.json"
            selection_path.write_text(
                json.dumps(
                    {
                        "baseline": {"weights": weights},
                        "selected": {"weights": weights, "members": members},
                    }
                )
            )
            confirmation = confirm_ensemble(
                config,
                paths,
                data,
                selection_path,
                root / "confirmation.json",
            )
            self.assertFalse(confirmation["passed"])
            self.assertEqual(confirmation["frozen_choice"], "baseline")


if __name__ == "__main__":
    unittest.main()
