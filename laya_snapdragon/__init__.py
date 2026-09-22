"""Laya typed decisions on the Snapdragon X NPU, via ONNX Runtime + QNN. No PyTorch at run time.

Prompt construction, calibration and the result schema follow upstream Laya exactly (vendored from
laya-mlx, see NOTICE); only the forward pass differs. Each question runs on the smallest compiled
NPU bucket it fits (sequence length x option count); anything larger falls back to the CPU model.
"""

import json
import re
import time
from pathlib import Path

import numpy as np

from .common import QTYPES, build_sequence, confidence_from_probs, render_options, temp_bucket
from .runtime import cpu_session, npu_session
from .tokenizer import Tokenizer

__version__ = "0.1.0"

HF_REPO = "convaiinnovations/laya"
HF_REVISION = "1c5edc17a7acd8701df6fc341c0d179f1c62c982"  # the checkpoint every number in the README was measured on
OUTPUTS = ["logits", "act"]
BUCKET = re.compile(r"laya_s(\d+)_m(\d+)_qnn_ctx\.onnx$")


def collate(items, pad_id, seq=None, markers=None):
    """Input tensors for ONNX. With seq/markers, pad to that fixed shape (the NPU needs static shapes);
    padded option slots are masked out, so the first k logits are unchanged."""
    n = len(items)
    length = seq or max(len(it["ids"]) for it in items)
    count = markers or max(2, max(len(it["markers"]) for it in items))
    b = {
        "input_ids": np.full((n, length), pad_id, dtype=np.int64),
        "attention_mask": np.zeros((n, length), dtype=np.int64),
        "marker_pos": np.zeros((n, count), dtype=np.int64),
        "marker_mask": np.zeros((n, count), dtype=np.bool_),
        "qtype": np.array([it["qtype"] for it in items], dtype=np.int64),
    }
    for i, it in enumerate(items):
        L, k = len(it["ids"]), len(it["markers"])
        b["input_ids"][i, :L] = it["ids"]
        b["attention_mask"][i, :L] = 1
        b["marker_pos"][i, :k] = it["markers"]
        b["marker_mask"][i, :k] = True
    return b


class Agent:
    """Drop-in for upstream `laya.Agent.predict(state, questions)`.

    device: "auto" (NPU buckets, CPU for what doesn't fit), "npu" (error if a question doesn't fit),
    or "cpu" (the dynamic fp32 model only; upstream-exact).
    """

    def __init__(self, models="models", device="auto", cpu_threads=None):
        models = Path(models)
        self.checkpoint = models / "laya"
        self.cfg = json.loads((self.checkpoint / "rl_agent_config.json").read_text())
        self.temperature = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options = self.cfg.get("temperature_by_options", {})
        self.tok = Tokenizer(self.checkpoint / "tokenizer")
        self.device = device
        self._cpu_model, self._cpu_threads, self._cpu = models / "onnx" / "laya_fp32.onnx", cpu_threads, None
        self.buckets = []  # (seq, markers, session), smallest first
        t = time.perf_counter()
        if device != "cpu":
            for ctx in sorted(models.glob("onnx/laya_s*_m*_qnn_ctx.onnx")):
                seq, markers = map(int, BUCKET.search(ctx.name).groups())
                self.buckets.append((seq, markers, npu_session(ctx)))
            self.buckets.sort(key=lambda b: (b[0], b[1]))
            if device == "npu" and not self.buckets:
                raise FileNotFoundError(f"no compiled NPU buckets in {models / 'onnx'}; run: python -m laya_snapdragon build")
        if device == "cpu" or not self.buckets:
            self._cpu_session()
        self.load_s = time.perf_counter() - t

    def _cpu_session(self):
        if self._cpu is None:
            if not self._cpu_model.exists():
                raise FileNotFoundError(f"{self._cpu_model} missing; run: python -m laya_snapdragon build")
            self._cpu = cpu_session(self._cpu_model, self._cpu_threads)
        return self._cpu

    @staticmethod
    def _to_internal(qdef):
        kind, criteria = qdef["type"], qdef.get("criteria")
        if kind == "choice" and isinstance(criteria, list):
            criteria = dict.fromkeys(criteria)
        ins = qdef["instructions"]
        return {"t": kind, "ins": ins if isinstance(ins, str) else json.dumps(ins), "crit": criteria}

    def prepare(self, state, questions):
        items, internal = [], []
        for qid, d in questions.items():
            q = self._to_internal(d)
            ids, markers = build_sequence(self.tok, state, q, self.cfg.get("max_len", 512), self.cfg.get("head_max_len", 192))
            if len(markers) != len(render_options(q)):
                raise ValueError(f"Question {qid!r} has too many options for the token budget")
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["t"]]})
            internal.append(q)
        return items, internal

    def route(self, item):
        """The bucket (seq, markers) an item runs on, or None for the CPU."""
        if self.device != "cpu":
            for seq, markers, _ in self.buckets:
                if len(item["ids"]) <= seq and len(item["markers"]) <= markers:
                    return seq, markers
        if self.device == "npu":
            raise ValueError(f"question needs {len(item['ids'])} tokens / {len(item['markers'])} options; no NPU bucket fits")
        return None

    def forward(self, items):
        """Raw (logits, act) per item, in order: NPU items one at a time, CPU leftovers as one batch."""
        out, cpu_rows = [None] * len(items), []
        sessions = {(s, m): sess for s, m, sess in self.buckets}
        for i, it in enumerate(items):
            r = self.route(it)
            if r is None:
                cpu_rows.append(i)
                continue
            logits, act = sessions[r].run(OUTPUTS, collate([it], self.tok.pad_token_id, *r))
            out[i] = (logits[0], act[0])
        if cpu_rows:
            batch = [items[i] for i in cpu_rows]
            logits, act = self._cpu_session().run(OUTPUTS, collate(batch, self.tok.pad_token_id))
            for row, i in enumerate(cpu_rows):
                out[i] = (logits[row], act[row])
        return out

    def predict(self, state, questions):
        items, internal = self.prepare(state, questions)
        answers = {}
        for qid, q, it, (logits, act) in zip(questions, internal, items, self.forward(items)):
            if not np.isfinite(logits).all() or not np.isfinite(act).all():
                raise FloatingPointError("Non-finite model outputs")
            act = np.exp(act - act.max()); act /= act.sum()
            k, qt = len(it["markers"]), it["qtype"]
            scale = self.temperature_by_options.get(temp_bucket(qt, k), self.temperature[qt])
            z = np.asarray(logits[:k], dtype=np.float64) / max(1e-3, float(scale))
            p = np.exp(z - z.max()); p /= p.sum()
            a = {"type": q["t"], "confidence": round(confidence_from_probs(p, k), 4),
                 "action": {"act_probability": round(float(act[0]), 4)}}
            if q["t"] == "choice":
                labels = list(q["crit"])
                a.update(choice=labels[int(p.argmax())],
                         probabilities={l: round(float(v), 4) for l, v in zip(labels, p)})
            elif q["t"] == "score":
                a.update(score=round(float((np.arange(k) * p).sum()), 4),
                         legend={str(i): v for i, v in enumerate(q["crit"])},
                         probabilities={str(i): round(float(v), 4) for i, v in enumerate(p)})
            else:
                a.update(noul=round(float(p[1]), 4), confidence=round(max(float(p[1]), 1.0 - float(p[1])), 4))
            answers[qid] = a
        return {"model": "laya-rl-agent", "answers": answers,
                "usage": {"input_tokens": sum(len(it["ids"]) for it in items), "output_tokens": 0}}
