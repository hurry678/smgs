"""Torch-side mechanism and delivery checks (server task T05).

These cover the properties the NumPy protocol tests cannot see: the actual
networks, the frozen text adapter cache, the optimiser mechanics and the
offline reload path.  Everything runs on CPU/FP32 so the result does not depend
on which GPU happens to be free.
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import data as D                                  # noqa: E402
import engine as E                                # noqa: E402
import models as M                                # noqa: E402
import optimizers as O                            # noqa: E402
from protocol_core import (gap_structure, interface_masks, make_missing,  # noqa: E402
                           publish_c2)

DEVICE = torch.device("cpu")
T = 50


def fixture(n=4, seed=7):
    """Small but protocol-valid batch: CLS + 8 tokens + SEP, padding after."""
    rng = np.random.default_rng(seed)
    tb = np.zeros((n, 3, T), dtype=np.int64)
    length = 10
    tb[:, 0, 0] = 101
    tb[:, 0, 1:length - 1] = rng.integers(1000, 2000, size=(n, length - 2))
    tb[:, 0, length - 1] = 102
    tb[:, 1, :length] = 1
    audio = rng.normal(size=(n, T, 74)).astype(np.float32)
    vision = rng.normal(size=(n, T, 35)).astype(np.float32)
    audio[:, length:] = 0.0
    vision[:, length:] = 0.0
    audio[0, 3:6] = 0.0                      # natural interior hole
    vision[1, :] = 0.0                       # a whole natively empty modality
    domain, observed = interface_masks(tb, audio, vision)
    return {"text_bert": tb, "audio": audio, "vision": vision,
            "domain": domain, "observed": observed}


def tensors(fx, index=None, device=DEVICE):
    idx = np.arange(len(fx["domain"])) if index is None else np.asarray(index)
    out = {k: torch.from_numpy(fx[k][idx]).to(device) for k in ("text_bert", "audio", "vision")}
    out["domain"] = torch.from_numpy(fx["domain"][idx]).to(device)
    out["observed"] = torch.from_numpy(fx["observed"][idx]).to(device)
    return out


def forward(model, t, visible=None):
    vis = t["observed"] if visible is None else visible
    return model(t["text_bert"], t["audio"], t["vision"], t["domain"], vis)


def stack(out):
    return torch.cat((out["logits"], out["raw_intensity"].unsqueeze(-1)), -1)


class StructureFeatureTest(unittest.TestCase):
    def test_matches_numpy_reference(self):
        fx = fixture()
        got = M.gap_structure_torch(torch.from_numpy(fx["domain"]),
                                    torch.from_numpy(fx["observed"])).numpy()
        want = gap_structure(fx["domain"], fx["observed"])
        self.assertEqual(got.shape, want.shape)
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)


class InvarianceTest(unittest.TestCase):
    def setUp(self):
        self.fx = fixture()
        torch.manual_seed(0)
        self.models = {"B2": M.build_model("three_modality_masked_mean_concat_mlp"),
                       "M2": M.build_model("bigru_domain_safe_gap_structure_gate", hidden=16),
                       "M3": M.build_model("M2_plus_local_cross_time_attention", hidden=16)}
        for model in self.models.values():
            model.to(DEVICE).eval()

    def test_unobserved_positions_never_leak(self):
        """O=0 positions may hold arbitrary finite placeholders; output is unchanged."""
        t = tensors(self.fx)
        poisoned = {k: v.clone() for k, v in t.items()}
        mask = ~self.fx["observed"]
        for mod, field in enumerate(("text_bert", "audio", "vision")):
            if field == "text_bert":
                continue
            sel = torch.from_numpy(mask[:, mod]).unsqueeze(-1).expand_as(poisoned[field])
            poisoned[field][sel] = 1e3
        with torch.no_grad():
            for name, model in self.models.items():
                base = stack(forward(model, t))
                other = stack(forward(model, poisoned))
                self.assertTrue(torch.isfinite(other).all(), name)
                self.assertLessEqual(float((base - other).abs().max()), 1e-5, name)

    def test_padding_values_do_not_change_semantics(self):
        t = tensors(self.fx)
        padded = {k: v.clone() for k, v in t.items()}
        outside = ~self.fx["domain"]
        wide = torch.from_numpy(outside).unsqueeze(-1)
        padded["audio"][wide.expand_as(padded["audio"])] = -5e2
        padded["vision"][wide.expand_as(padded["vision"])] = 5e2
        # text_bert is [B,3,T]: the time mask must broadcast as [B,1,T], not [B,T,1]
        # D is strictly between CLS and SEP (docs/数据与缺失协议.md), and CLS/SEP stay as
        # encoder support rather than padding, so only genuine out-of-domain PAD tokens
        # are poisoned here; poisoning the specials would be a contract violation, not
        # a padding leak.
        tokens = self.fx["text_bert"][:, 0]
        pad = outside & ~((tokens == 101) | (tokens == 102))
        padded["text_bert"][torch.from_numpy(pad).unsqueeze(1)
                            .expand_as(padded["text_bert"])] = 1234
        with torch.no_grad():
            for name, model in self.models.items():
                base = stack(forward(model, t))
                other = stack(forward(model, padded))
                self.assertLessEqual(float((base - other).abs().max()), 1e-6, name)

    def test_all_empty_sample_stays_finite(self):
        fx = fixture()
        fx["observed"][:] = False
        t = tensors(fx)
        with torch.no_grad():
            for name, model in self.models.items():
                out = forward(model, t)
                self.assertTrue(torch.isfinite(out["logits"]).all(), name)
                self.assertTrue(torch.isfinite(out["raw_intensity"]).all(), name)
                self.assertTrue(bool(out["diagnostics"]["no_evidence"].all()), name)

    def test_native_empty_modality_is_kept(self):
        fx = fixture()
        t = tensors(fx)
        self.assertFalse(fx["observed"][1, 2].any())
        with torch.no_grad():
            out = forward(self.models["B2"], t)
        self.assertTrue(torch.isfinite(out["logits"]).all())


class TextAdapterTest(unittest.TestCase):
    def setUp(self):
        self.fx = fixture()
        torch.manual_seed(0)
        self.model = M.build_model("three_modality_masked_mean_concat_mlp").to(DEVICE).eval()
        self.t = tensors(self.fx)

    def _capture(self, visible):
        seen = {}

        def hook(module, args, kwargs):
            # BertModel is called with keyword arguments, so a plain pre-hook sees an
            # empty positional tuple; capture the real call instead
            ids = kwargs.get("input_ids", args[0] if args else None)
            attn = kwargs.get("attention_mask", args[1] if len(args) > 1 else None)
            seen["input_ids"] = ids.detach().clone()
            seen["attention"] = attn.detach().clone()

        handle = self.model.text_encoder.bert.register_forward_pre_hook(hook, with_kwargs=True)
        with torch.no_grad():
            self.model.text_encoder(self.t["text_bert"], self.t["domain"], visible)
        handle.remove()
        return seen

    def test_interior_hole_is_masked_before_encoding(self):
        hidden_pos = 4
        hole = self.t["observed"].clone()
        hole[0, 0, hidden_pos] = False
        seen = self._capture(hole[:, 0])
        self.assertEqual(int(seen["input_ids"][0, hidden_pos]), 103)
        self.assertEqual(int(seen["attention"][0, hidden_pos]), 0)
        self.assertTrue(bool(self.fx["domain"][0, hidden_pos]))

    def test_cache_never_reuses_the_full_text_hidden_state(self):
        full = self.t["observed"]
        masked = full.clone()
        masked[0, 0, 4] = False
        E.set_text_cache(self.model, "full")
        with torch.no_grad():
            first = self.model.encode_text(self.t["text_bert"], self.t["domain"], full[:, 0])
            again = self.model.encode_text(self.t["text_bert"], self.t["domain"], full[:, 0])
            other = self.model.encode_text(self.t["text_bert"], self.t["domain"], masked[:, 0])
        self.assertIs(first, again)
        self.assertGreater(float((first - other).abs().max()), 1e-6)
        self.assertEqual(float(other[0, 4].abs().max()), 0.0)
        with torch.no_grad():
            out_full = stack(forward(self.model, self.t, full))
            out_masked = stack(forward(self.model, self.t, masked))
        self.assertGreater(float((out_full - out_masked).abs().max()), 1e-6)

    def test_bert_stays_eval_and_frozen(self):
        before = [p.detach().clone() for p in self.model.text_encoder.parameters()]
        self.model.train()
        self.assertFalse(self.model.text_encoder.bert.training)
        out = forward(self.model, self.t)
        loss = out["logits"].sum() + out["raw_intensity"].sum()
        loss.backward()
        after = list(self.model.text_encoder.parameters())
        for old, new in zip(before, after):
            self.assertIsNone(new.grad)
            self.assertTrue(torch.equal(old, new.detach()))
        self.assertTrue(any(p.grad is not None for p in self.model.head.parameters()))


class NormalizerTest(unittest.TestCase):
    def test_train_only_fit_and_zero_restore(self):
        train = fixture(n=6, seed=3)
        valid = fixture(n=4, seed=11)
        stats = D.fit_normalizer(train["audio"], train["vision"], train["observed"])
        for name in ("audio", "vision"):
            mean, std = stats[name]["mean"], stats[name]["std"]
            self.assertTrue((std >= 1e-5).all())
            values = D.apply_normalizer(valid[name], valid["observed"][:, {"audio": 1,
                                                                          "vision": 2}[name]],
                                        mean, std)
            zero = ~valid["observed"][:, {"audio": 1, "vision": 2}[name]]
            self.assertEqual(float(np.abs(values[zero]).max()), 0.0)
            self.assertTrue(np.isfinite(values).all())
        refit = D.fit_normalizer(train["audio"], train["vision"], train["observed"])
        for name in ("audio", "vision"):
            np.testing.assert_allclose(refit[name]["mean"], stats[name]["mean"])
            np.testing.assert_allclose(refit[name]["std"], stats[name]["std"])


class MaskBankTest(unittest.TestCase):
    def test_reordering_reproduces_sample_digests(self):
        cfg = json.loads((ROOT / "configs" / "protocol.json").read_text())
        fx = fixture(n=6, seed=5)
        ids = [f"videoA${i}" for i in range(6)]
        first = D.build_bank("select", fx["domain"], fx["observed"], ids, cfg)
        repeat = D.build_bank("select", fx["domain"], fx["observed"], ids, cfg)
        self.assertEqual(first["bank_sha256"], repeat["bank_sha256"])
        order = np.array([3, 0, 5, 1, 4, 2])
        second = D.build_bank("select", fx["domain"][order], fx["observed"][order],
                              [ids[i] for i in order], cfg)
        for c in range(len(first["condition_ids"])):
            for i, sid in enumerate(ids):
                j = int(np.flatnonzero(order == i)[0])
                self.assertEqual(first["sample_sha256"][c][i], second["sample_sha256"][c][j])
        self.assertFalse(bool((D.unpack_masks(first["packed"][0], T) & ~fx["observed"]).any()))


class OptimizerTest(unittest.TestCase):
    def test_pcgrad_projection_removes_conflict(self):
        a = [torch.tensor([1.0, 0.0])]
        b = [torch.tensor([-1.0, 1.0])]
        ga, gb = O.pcgrad_project(a, b)
        # the projected pair must be free of the original conflict, and must not
        # collapse the regression direction onto the classification gradient
        self.assertLessEqual(abs(float((ga[0] * b[0]).sum())), 1e-6)
        self.assertLessEqual(abs(float((gb[0] * a[0]).sum())), 1e-6)
        self.assertTrue(torch.allclose(ga[0], torch.tensor([0.5, 0.5]), atol=1e-6))
        self.assertTrue(torch.allclose(gb[0], torch.tensor([0.0, 1.0]), atol=1e-6))

    def test_sam_perturbs_and_rolls_back(self):
        torch.manual_seed(0)
        layer = torch.nn.Linear(4, 2)
        base = torch.optim.SGD(layer.parameters(), lr=0.1)
        sam = O.SAM(layer.parameters(), base, rho=0.05)
        x = torch.randn(8, 4)
        out = layer(x)
        loss = out.pow(2).mean()
        base.zero_grad(set_to_none=True)
        loss.backward()
        grads = [p.grad.detach().clone() for p in layer.parameters()]
        before = [p.detach().clone() for p in layer.parameters()]
        sam.first_step()
        perturbed = [p.detach().clone() for p in layer.parameters()]
        self.assertGreater(float(max((q - p).abs().max() for p, q in zip(before, perturbed))), 0.0)
        sam.second_step()
        after = [p.detach().clone() for p in layer.parameters()]
        for old, new, grad in zip(before, after, grads):
            self.assertTrue(torch.allclose(old - 0.1 * grad, new, atol=1e-6))
        self.assertIsNone(sam._e_w)


class CheckpointTest(unittest.TestCase):
    def test_offline_reload_matches(self):
        fx = fixture()
        t = tensors(fx)
        torch.manual_seed(1)
        model = M.build_model("bigru_binary_mask_gate_time_pool", hidden=16).to(DEVICE).eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            sha = E.save_checkpoint(path, model, {"run_id": "unit"})
            self.assertEqual(len(sha), 64)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertFalse(any(k.startswith("text_encoder.") for k in payload["state_dict"]))
            fresh = M.build_model("bigru_binary_mask_gate_time_pool", hidden=16).to(DEVICE).eval()
            E.load_checkpoint(path, fresh)
            with torch.no_grad():
                a = stack(forward(model, t))
                b = stack(forward(fresh, t))
            self.assertLessEqual(float((a - b).abs().max()), 1e-6)


class PublicationTest(unittest.TestCase):
    def test_publish_rule_and_readback(self):
        prob = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        raw = np.array([-5.0, 2.5, 0.9])
        cls, intensity = publish_c2(prob, raw, epsilon=1e-4)
        self.assertEqual(list(cls), [0, 1, 2])
        self.assertLess(intensity[0], 0)
        self.assertEqual(intensity[1], 0.0)
        self.assertGreater(intensity[2], 0)
        self.assertTrue((np.abs(intensity) <= 3).all())
        self.assertAlmostEqual(float(intensity[0]), -3.0, places=6)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "q2_predictions.csv"
            labels = ("Negative", "Neutral", "Positive")
            with path.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["sample_id", "原文件名", "polarity", "intensity"])
                for i in range(30):
                    writer.writerow([f"附件3_{i + 1:02d}", f"附件3_{i + 1:02d}.pkl",
                                     labels[i % 3], f"{intensity[i % 3]:.6f}"])
            rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
            self.assertEqual(len(rows), 30)
            self.assertEqual([r["sample_id"] for r in rows],
                             [f"附件3_{i:02d}" for i in range(1, 31)])
            self.assertEqual(rows[0]["polarity"], "Negative")
            self.assertLessEqual(abs(float(rows[0]["intensity"]) + 3.0), 5e-7)

    def test_make_missing_never_restores_observation(self):
        fx = fixture()
        ids = [f"videoB${i}" for i in range(len(fx["domain"]))]
        vis, rows = make_missing(fx["domain"], fx["observed"], ids, (1,), 0.3,
                                 seed=1, split="valid", placement="middle")
        self.assertFalse(bool((vis & ~fx["observed"]).any()))
        self.assertTrue(rows)
        self.assertTrue(all(0 < r["visible"] < r["original_observed"] for r in rows))


if __name__ == "__main__":
    unittest.main()