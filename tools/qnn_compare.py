"""Compare QNN compilers: compile one fixed-shape model with the installed onnxruntime-qnn, or time
EP-context files with the same fixed input.

    python tools/qnn_compare.py compile models/onnx/laya_s128_m8.onnx ctx_a.onnx
    python tools/qnn_compare.py time ctx_a.onnx          # time one context per process

Works with onnxruntime-qnn 1.24.x (built-in EP) and 2.x (plugin EP)."""
import sys, time
import numpy as np, onnxruntime as ort
OPTS = {"enable_htp_fp16_precision": "1", "htp_performance_mode": "burst"}

def session(path, ctx_out=None):
    so = ort.SessionOptions(); so.log_severity_level = 3
    if ctx_out:
        so.add_session_config_entry("ep.context_enable", "1"); so.add_session_config_entry("ep.context_file_path", ctx_out)
    if "QNNExecutionProvider" in ort.get_available_providers():
        return ort.InferenceSession(path, so, providers=[("QNNExecutionProvider", {"backend_path": "QnnHtp.dll", **OPTS}), "CPUExecutionProvider"])
    import onnxruntime_qnn as oq
    if not getattr(session, "registered", False):
        ort.register_execution_provider_library(oq.EP_NAME, oq.get_library_path()); session.registered = True
    so.add_provider_for_devices([d for d in ort.get_ep_devices() if d.ep_name == oq.EP_NAME and d.device.type == ort.OrtHardwareDeviceType.NPU], OPTS)
    return ort.InferenceSession(path, so)

rng = np.random.default_rng(0)
FEED = {"input_ids": rng.integers(5, 30000, (1, 128)).astype(np.int64), "attention_mask": np.ones((1, 128), np.int64),
        "marker_pos": (np.arange(8, dtype=np.int64) * 3 + 10)[None], "marker_mask": np.ones((1, 8), np.bool_), "qtype": np.array([0], np.int64)}
if sys.argv[1] == "compile":
    t = time.perf_counter(); session(sys.argv[2], sys.argv[3]); print(f"compiled {sys.argv[3]} in {time.perf_counter()-t:.0f}s")
else:
    ss = {p: session(p) for p in sys.argv[2:]}
    for s in ss.values():
        for _ in range(5): s.run(None, FEED)
    lat = {p: [] for p in ss}
    for _ in range(10):                      # 10 rounds x 20 runs, alternating, so load/thermal drift hits both equally
        for p, s in ss.items():
            for _ in range(20):
                t = time.perf_counter(); s.run(None, FEED); lat[p].append(time.perf_counter() - t)
    for p, v in lat.items():
        v = np.array(v) * 1000; print(f"{p}: P50 {np.percentile(v,50):.1f} ms  P90 {np.percentile(v,90):.1f} ms  min {v.min():.1f} ms  (n={len(v)})")
