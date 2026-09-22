# laya-snapdragon

Run [Laya](https://github.com/NandhaKishorM/laya), the calibrated typed-decision model from Convai
Innovations, on the **Snapdragon X NPU** of a Windows on ARM laptop. You get the same answers as
upstream PyTorch Laya, roughly 5× faster than the CPU, and no PyTorch at run time.

Laya doesn't generate text. It reads a *state* (an email, a ticket, a JSON document) and answers typed
questions about it in one forward pass: `choice` (pick an option), `score` (a place on a rubric), and
`noul` (a calibrated yes/no probability). That's routing, triage, moderation and gating, with real
probabilities.

| end to end (`python -m laya_snapdragon bench`) | NPU | CPU, native ARM64 fp32 |
|---|---|---|
| 1 question, short email | **17 ms** | 97 ms |
| 3 questions, short email | **52 ms** | 272 ms |
| 3 questions, 512-token document | **677 ms** | 2.6 s |

Measured on a Snapdragon X2 Elite (X2-90 GPU, Hexagon HTP v81), 32 GB, Windows 11 on AC power with an
idle machine. Background load moves the NPU numbers noticeably (we saw 17–40 ms for the same call), so
expect a range.

## Quick start

Needs **native ARM64 Python 3.11–3.13**. The x64 python.org installer runs under emulation and gets
x64 wheels. Check with `python -c "import sysconfig; print(sysconfig.get_platform())"`: it must print
`win-arm64`.

```powershell
git clone https://github.com/piffie/laya-snapdragon; cd laya-snapdragon
py -V:3.13-arm64 -m venv .venv            # or the full path to your ARM64 python.exe
.venv\Scripts\activate

pip install -e .
pip install torch --index-url https://download.pytorch.org/whl/cpu   # native win-arm64 wheel
$env:PYTHONUTF8 = "1"                     # upstream Laya's setup.py reads its README as cp1252 otherwise
pip install -e ".[export]"

python -m laya_snapdragon build           # ~10 min once: download 0.8 GB, export, compile 3 NPU buckets
python -m laya_snapdragon verify --torch  # NPU vs CPU vs PyTorch on built-in examples
python -m laya_snapdragon bench
```

Disk: about 9 GB under `models/` (the fp32 ONNX plus a fixed-shape copy and a compiled context per
bucket). After `build` you can uninstall torch; running only needs `onnxruntime-qnn`, `tokenizers` and `numpy`.

```python
from laya_snapdragon import Agent

agent = Agent("models")                   # loads the compiled NPU buckets in ~5 s
result = agent.predict(
    {"subject": "Duplicate charge on invoice #4411",
     "body": "We were billed twice for March. Refund the duplicate today or we cancel."},
    {
        "department": {"type": "choice", "instructions": "Which department should handle this email?",
                       "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages",
                                    "sales": "pricing, contracts", "other": "everything else"}},
        "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
    },
)
print(result["answers"]["department"]["choice"], result["answers"]["churn_risk"]["noul"])
```

`Agent.predict` returns the same structure as upstream `laya.Agent.predict`. From the shell:
`python -m laya_snapdragon predict --state "..." --questions questions.json`.

## How it works

The NPU (Qualcomm HTP, reached through ONNX Runtime's QNN execution provider) only runs graphs with
static shapes, and only runs a subset of ONNX ops well. So:

1. **Export with dynamic shapes** (`torch.onnx.export(dynamo=True, dynamic_shapes=...)`). A plain
   trace bakes the sequence length into a Reshape inside ModernBERT's attention. The existing community
   ONNX export has exactly that problem: it only runs at batch 1 × 512 tokens.
2. **Buckets.** `build` pins the model to batch 1 × {128, 256, 512} tokens × 8 options and compiles
   each shape once into a cached QNN context (`*_qnn_ctx.onnx` + `.bin`, ~850 MB each, 1–2 min).
   At run time each question goes to the smallest bucket it fits. Padding is masked out, so the
   real option logits are unchanged. Questions with more than 8 options, or longer than the largest
   bucket you built, fall back to the CPU model.
3. **Two exact graph rewrites** (`build.npu_rewrite`). Without them the HTP runs the model cut into ~60
   pieces with CPU round trips between them, and it's *slower than the CPU* (156 ms P50, 673 ms P95 for
   one question):
   - `IsNaN(x)` → `Not(Equal(x, x))`, 28 times
   - erf-GELU (`0.5·x·(1+Erf(x/√2))`) → one `Gelu` op via ONNX Runtime's own fusion, 30 times

   After them the graph is 2 NPU partitions plus one `GatherND` on the CPU (the option-marker gather,
   which costs nothing).
4. **fp16 on the HTP** (`enable_htp_fp16_precision`), in burst performance mode.

## Fidelity

- **CPU model (fp32 ONNX) vs upstream PyTorch** on laya-mlx's 63-question parity fixtures (8
  languages, up to 512 tokens and 20 options):
  - same argmax 63/63
  - max calibrated-probability difference 2.5e-6
  - tokens identical
  - public result identical in all 16 cases
- **NPU (fp16) vs PyTorch**, same fixtures, the questions that fit a bucket:
  - same argmax 59/59
  - max probability difference 5.0e-3 (128) and 6.7e-3 (256)
  - `verify` on the built-in examples: 12/12 same answers, max 1.3e-2 at 512 tokens

fp16 changes probabilities in the second or third decimal. If you need byte-identical results with
upstream, use `Agent(device="cpu")`.

## Things we learned the hard way (Windows on ARM)

- **Check that your Python is really ARM64.** An x64 Python works: the NPU still runs, through x64
  wheels under emulation. But CPU numbers are wrong, and the `onnxruntime-qnn` 2.x plugin then fails
  with `No libraries available for architecture: arm64`.
- **`onnxruntime-qnn==1.24.4` is pinned on purpose.** The 2.x line (the plugin EP, QNN SDK 2.50) works
  natively. It gives identical answers, but its compiled graphs ran about 1.5× slower here than 1.24.4's
  (QNN SDK 2.42), measured side by side (52–64 ms vs 35–38 ms under the same load). The runtime code
  (`runtime.py`) supports both, so upgrading is a one-line change once that's fixed.
- **The 512 bucket spills.** The compiler reports ~4 GB of VTCM spill traffic at 512 tokens, so it costs
  ~225 ms per question against 17 ms at 128. If your inputs are short, build `--seq 128 256`.
- The community `tozp/laya-onnx` builds: fp32 is fixed at 512 tokens, fp16 crashes or fails ORT's type
  check, and int8 flips about a third of the answers (40/62 argmax agreement). Build your own instead.
- Batch 1 only on the NPU; multi-question calls run sequentially (about 17 ms each at 128).

## Status

This is an experiment, not a product. It's tested on one machine (Snapdragon X2 Elite). The package
also ships HTP v68/v73 libraries, so Snapdragon X Elite / X Plus (v73) should work, but that's untested.
Reports welcome. Only the English `convaiinnovations/laya` checkpoint is wired up; the multilingual one
(mmBERT, 1,024 context) should need only a different download and bucket sizes.

## Credits and license

Apache-2.0. Laya and its weights are by [Convai Innovations](https://huggingface.co/convaiinnovations/laya).
The prompt construction, calibration and tokenizer code are vendored from
[mizorewww/laya-mlx](https://github.com/mizorewww/laya-mlx), which also supplied the parity fixtures
we measured against. See [NOTICE](NOTICE).
