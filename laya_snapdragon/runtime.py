"""ONNX Runtime sessions: the Snapdragon NPU (QNN HTP) and a CPU fallback."""

import os
from pathlib import Path

import onnxruntime as ort

# fp16 on the HTP; burst keeps the NPU clocked up between single requests
QNN_OPTIONS = {"enable_htp_fp16_precision": "1", "htp_performance_mode": "burst"}


def qnn_mode():
    """'builtin' (onnxruntime-qnn 1.x ships the QNN EP inside onnxruntime), 'plugin' (2.x), or None."""
    if "QNNExecutionProvider" in ort.get_available_providers():
        return "builtin"
    try:
        import onnxruntime_qnn  # noqa: F401
    except ImportError:
        return None
    return "plugin" if hasattr(ort, "register_execution_provider_library") else None


_registered = False


def _npu_devices():
    global _registered
    import onnxruntime_qnn as oq

    if not _registered:
        ort.register_execution_provider_library(oq.EP_NAME, oq.get_library_path())
        _registered = True
    return [d for d in ort.get_ep_devices()
            if d.ep_name == oq.EP_NAME and d.device.type == ort.OrtHardwareDeviceType.NPU]


def npu_session(model, context=None, options=None):
    """A QNN HTP session. With `context`, compile `model` once and cache the result there
    (ONNX Runtime's EP context); later calls load `context` directly, which takes seconds."""
    model, context = Path(model), Path(context) if context else None
    opts = {**QNN_OPTIONS, **(options or {})}
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.add_session_config_entry("session.disable_cpu_ep_fallback", "0")  # GatherND stays on CPU
    source = model
    if context is not None:
        if context.exists():
            source = context
        else:
            so.add_session_config_entry("ep.context_enable", "1")
            so.add_session_config_entry("ep.context_file_path", str(context))
    mode = qnn_mode()
    if mode == "builtin":
        providers = [("QNNExecutionProvider", {"backend_path": "QnnHtp.dll", **opts}), "CPUExecutionProvider"]
        return ort.InferenceSession(str(source), so, providers=providers)
    if mode == "plugin":
        devices = _npu_devices()
        if not devices:
            raise RuntimeError("onnxruntime-qnn found no NPU on this machine")
        so.add_provider_for_devices(devices, opts)
        return ort.InferenceSession(str(source), so)
    raise RuntimeError("No QNN execution provider: pip install onnxruntime-qnn (on Windows on Snapdragon)")


def cpu_session(model, threads=None):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = threads or min(8, os.cpu_count() or 1)
    return ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])
