"""Build the ONNX models and compile the NPU buckets.

    python -m laya_snapdragon build                  # everything, default buckets 128 256 512 (x 8 options)
    python -m laya_snapdragon build --seq 128 256    # fewer / other buckets

Steps (each is skipped when its output already exists):
1. download  the Laya checkpoint from Hugging Face, pinned revision
2. export    PyTorch -> ONNX with dynamic batch/sequence/option dims       (needs the [export] extras)
3. bucket    pin one shape per bucket and rewrite 3 op types the NPU can't run (exact rewrites)
4. compile   QNN HTP fp16 context per bucket, cached next to the model     (~1-2 min each, once)
"""

import collections
import sys
import time
from pathlib import Path

from . import HF_REPO, HF_REVISION

INPUTS = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]


def log(msg):
    print(msg, flush=True)


def download(models: Path):
    from huggingface_hub import snapshot_download

    dst = models / "laya"
    if (dst / "model.safetensors").exists():
        return dst
    log(f"download  {HF_REPO}@{HF_REVISION[:7]} -> {dst}")
    snapshot_download(HF_REPO, revision=HF_REVISION, local_dir=str(dst),
                      allow_patterns=["model.safetensors", "rl_agent_config.json", "encoder/*", "tokenizer/*"])
    return dst


def export(models: Path):
    """torch.export keeps batch/sequence/option dims symbolic. A plain trace bakes the sequence
    length into a Reshape inside ModernBERT's attention, so the model then only runs at that length."""
    target = models / "onnx" / "laya_fp32.onnx"
    if target.exists():
        return target
    try:
        import torch
        from laya.agent import Agent as Reference
    except ImportError:
        sys.exit("export needs PyTorch and upstream Laya: see 'Install' in the README (the [export] extras)")
    target.parent.mkdir(parents=True, exist_ok=True)
    log("export    PyTorch -> ONNX (fp32, dynamic shapes; a few minutes, ~6 GB RAM)")
    t = time.perf_counter()
    model = Reference(str(models / "laya"), device="cpu").model.eval()
    B, L = 3, 96
    dummy = (
        torch.randint(5, 30000, (B, L), dtype=torch.long),
        torch.ones((B, L), dtype=torch.long),
        torch.tensor([[40, 50, 60, 70]] * B, dtype=torch.long),
        torch.ones((B, 4), dtype=torch.bool),
        torch.tensor([0, 1, 2], dtype=torch.long),
    )
    batch = torch.export.Dim("batch", min=1, max=128)
    seq = torch.export.Dim("seq", min=8, max=1024)
    markers = torch.export.Dim("markers", min=2, max=64)
    shapes = {"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq},
              "marker_pos": {0: batch, 1: markers}, "marker_mask": {0: batch, 1: markers}, "qtype": {0: batch}}
    with torch.inference_mode():
        program = torch.onnx.export(model, dummy, dynamo=True, input_names=INPUTS, output_names=["logits", "act"],
                                    dynamic_shapes=shapes, opset_version=18, optimize=True)
    program.save(str(target), external_data=True)
    log(f"          {target} ({time.perf_counter() - t:.0f}s)")
    return target


def npu_rewrite(model):
    """Make the graph run on the HTP in one piece without changing what it computes.

    - IsNaN(x) -> Not(Equal(x, x)): exact, NaN is the only value unequal to itself.
    - erf-GELU (0.5*x*(1+Erf(x/sqrt2))) -> one Gelu op, via ONNX Runtime's own fusion.
    GatherND (the option-marker gather at the head) is left on the CPU; it costs nothing.
    Without this the HTP graph is cut into ~60 pieces and runs slower than the CPU.
    """
    from onnx import helper
    from onnxruntime.transformers.fusion_gelu import FusionGelu
    from onnxruntime.transformers.onnx_model import OnnxModel

    g = model.graph
    nodes = []
    for n in g.node:
        if n.op_type == "IsNaN":
            eq = n.output[0] + "__selfeq"
            nodes.append(helper.make_node("Equal", [n.input[0], n.input[0]], [eq], name=n.name + "_eq"))
            nodes.append(helper.make_node("Not", [eq], [n.output[0]], name=n.name + "_not"))
        else:
            nodes.append(n)
    del g.node[:]
    g.node.extend(nodes)
    om = OnnxModel(model)
    FusionGelu(om).apply()
    om.prune_graph()
    om.topological_sort()
    left = collections.Counter(n.op_type for n in om.model.graph.node if n.op_type in ("Erf", "IsNaN"))
    if left:
        raise RuntimeError(f"NPU rewrite incomplete, still in graph: {dict(left)}")
    return om.model


def bucket(models: Path, seq: int, markers: int):
    import onnx
    from onnxruntime.tools.onnx_model_utils import fix_output_shapes, make_dim_param_fixed

    target = models / "onnx" / f"laya_s{seq}_m{markers}.onnx"
    if target.exists():
        return target
    log(f"bucket    {seq} tokens x {markers} options")
    m = onnx.load(str(models / "onnx" / "laya_fp32.onnx"))
    for dim, value in (("batch", 1), ("seq", seq), ("markers", markers)):
        make_dim_param_fixed(m.graph, dim, value)
    fix_output_shapes(m)
    m = npu_rewrite(m)
    onnx.save(m, str(target), save_as_external_data=True, location=target.name + ".data")
    return target


def compile_bucket(models: Path, seq: int, markers: int):
    from .runtime import npu_session

    ctx = models / "onnx" / f"laya_s{seq}_m{markers}_qnn_ctx.onnx"
    if ctx.exists():
        return ctx
    log(f"compile   {seq} x {markers} for the NPU (QNN HTP, fp16; a minute or two, once)")
    t = time.perf_counter()
    npu_session(models / "onnx" / f"laya_s{seq}_m{markers}.onnx", context=ctx)
    log(f"          {ctx} ({time.perf_counter() - t:.0f}s)")
    return ctx


def build(models="models", seqs=(128, 256, 512), markers=8, npu=True):
    models = Path(models)
    download(models)
    export(models)
    if npu:
        for seq in seqs:
            bucket(models, seq, markers)
            compile_bucket(models, seq, markers)
    log("done")
