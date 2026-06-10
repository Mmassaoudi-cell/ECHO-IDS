from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from echo_ids_model import (
    ECHOConfig,
    ECHOSystem,
    EGSINdexVerifier,
    EchoIDS,
    EvidenceBuilder,
)


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"edge_iiot", "rt_iot2022", "unsw_nb15"}
SEEDS = {17, 29, 43}


class ReproducibilityPackageTests(unittest.TestCase):
    def test_unified_architecture_forward_and_evidence(self) -> None:
        config = ECHOConfig(
            input_features=6,
            graph_features=3,
            attack_classes=4,
            hidden_dim=16,
            attention_heads=4,
        )
        model = EchoIDS(config).eval()
        sequence = np.random.default_rng(17).normal(size=(4, 6)).astype("float32")
        graph = np.random.default_rng(29).normal(size=(3, 3)).astype("float32")
        output = model(
            torch.from_numpy(sequence).unsqueeze(0),
            torch.from_numpy(graph).unsqueeze(0),
        )
        self.assertEqual(tuple(output.binary_logits.shape), (1, 2))
        self.assertEqual(tuple(output.multiclass_logits.shape), (1, 4))
        builder = EvidenceBuilder(
            [f"feature_{index}" for index in range(6)],
            {0: "Normal", 1: "DDoS", 2: "Recon", 3: "Backdoor"},
        )
        system = ECHOSystem(model, builder)
        evidence, _ = system.detect_and_build_evidence(
            dataset="synthetic",
            sample_id=1,
            sequence=sequence,
            graph_nodes=graph,
        )
        self.assertIn("text_summary", evidence)
        self.assertEqual(len(evidence["evidence_vector"]), config.hidden_dim)

    def test_unified_verifier_returns_field_explanations(self) -> None:
        class DummyEncoder:
            def encode(self, sentences, **kwargs):
                vectors = []
                for sentence in sentences:
                    seed = sum(map(ord, sentence)) % (2**32)
                    vector = np.random.default_rng(seed).normal(size=12)
                    vectors.append(vector / np.linalg.norm(vector))
                return np.asarray(vectors)

        evidence = {
            "text_summary": "DDoS over tcp from client to service endpoint.",
            "attack_type": "DDoS",
            "protocol": "tcp",
            "source_role": "client",
            "destination_role": "service endpoint",
            "port_group": "registered",
            "severity_score": 3,
        }
        verifier = EGSINdexVerifier(DummyEncoder())
        result = verifier.verify(
            evidence,
            "DDoS over udp caused an emergency shutdown.",
            [
                "DDoS over tcp from client to service endpoint.",
                "DDoS over udp caused an emergency shutdown.",
            ],
        )
        self.assertGreater(result.unsupported_claim_penalty, 0)
        self.assertIn("physical_consequence", result.contradicted_fields)

    def test_package_excludes_paper_rendering_code(self) -> None:
        self.assertFalse((ROOT / "figures").exists())
        self.assertFalse((ROOT / "tables").exists())
        self.assertFalse((ROOT / "manuscript").exists())
        source_names = {path.name for path in ROOT.rglob("*.py")}
        self.assertNotIn("make_figures.py", source_names)
        self.assertNotIn("make_tables.py", source_names)

    def test_required_method_modules_exist(self) -> None:
        required = (
            "preprocessing/preprocess_ids.py",
            "preprocessing/audit_splits.py",
            "models/ids_models.py",
            "models/eg_sindex.py",
            "training/train_ids.py",
            "training/evaluate_hybrid_echo.py",
            "training/evaluate_hallucination.py",
            "evidence_builder/build_evidence.py",
            "llm_reports/generate_cyber_reports.py",
            "tuning/tune_echo_ids.py",
            "tuning/tune_eg_sindex.py",
            "ablations/run_ids_ablation.py",
            "ablations/evaluate_score_ablations.py",
            "ablations/run_sensitivity.py",
        )
        for relative in required:
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_committed_split_manifests_pass(self) -> None:
        for dataset in DATASETS:
            path = ROOT / "Data/processed" / dataset / "split_manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "PASS")
            self.assertEqual(manifest["overlap"]["train_val"], 0)
            self.assertEqual(manifest["overlap"]["train_test"], 0)
            self.assertEqual(manifest["overlap"]["val_test"], 0)
            self.assertTrue(manifest["test_not_used_for_fit_or_thresholds"])
            self.assertEqual(set(manifest["partitions"]), {"train", "val", "test"})

    def test_reported_runs_use_all_fixed_seeds(self) -> None:
        checks = (
            ("results/ids_binary_runs.csv", ("dataset", "model")),
            ("results/ids_multiclass_runs.csv", ("dataset", "model")),
            ("results/ids_binary_hybrid_runs.csv", ("dataset", "model")),
            ("results/hallucination_runs.csv", ("method",)),
            ("results/joint_pipeline_runs.csv", ("dataset",)),
        )
        for relative, group_columns in checks:
            frame = pd.read_csv(ROOT / relative)
            for _, group in frame.groupby(list(group_columns)):
                self.assertEqual(set(group["seed"].astype(int)), SEEDS, relative)

    def test_reported_dataset_coverage(self) -> None:
        binary = pd.read_csv(ROOT / "results/ids_binary_runs.csv")
        multiclass = pd.read_csv(ROOT / "results/ids_multiclass_runs.csv")
        self.assertEqual(set(binary["dataset"]), DATASETS)
        self.assertEqual(set(multiclass["dataset"]), DATASETS)

    def test_validation_only_tuning_is_recorded(self) -> None:
        ids = json.loads(
            (ROOT / "tuning/echo_ids_best.json").read_text(encoding="utf-8")
        )
        self.assertFalse(ids["test_accessed"])
        hallucination = json.loads(
            (ROOT / "tuning/eg_sindex_best.json").read_text(encoding="utf-8")
        )
        self.assertEqual({entry["seed"] for entry in hallucination}, SEEDS)
        self.assertTrue(
            all(not entry["test_accessed_during_tuning"] for entry in hallucination)
        )


if __name__ == "__main__":
    unittest.main()
