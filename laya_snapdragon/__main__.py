"""python -m laya_snapdragon {build,predict,verify,bench} --help"""

import argparse
import json
import sys
import time

import numpy as np


def cmd_build(a):
    from .build import build

    build(a.models, a.seq, a.markers, npu=not a.no_npu)


def cmd_predict(a):
    from . import Agent

    state = a.state if a.state is not None else sys.stdin.read()
    try:
        state = json.loads(state)
    except ValueError:
        pass  # plain text state
    questions = json.loads(open(a.questions, encoding="utf-8").read())
    agent = Agent(a.models, device=a.device)
    print(json.dumps(agent.predict(state, questions), indent=2, ensure_ascii=False))


def _probs(agent, state, questions):
    """Calibrated probabilities per question, via the public result."""
    ans = agent.predict(state, questions)["answers"]
    out = {}
    for qid, a in ans.items():
        out[qid] = np.array([a["noul"]]) if a["type"] == "noul" else np.array(list(a["probabilities"].values()))
    return out


def cmd_verify(a):
    """NPU vs the upstream-exact CPU model (and vs PyTorch, with --torch) on built-in examples."""
    from . import Agent
    from .examples import CASES

    npu, cpu = Agent(a.models, device="auto"), Agent(a.models, device="cpu")
    ref = None
    if a.torch:
        from laya.agent import Agent as Reference  # upstream Laya, the [export] extras

        ref = Reference(str(cpu.checkpoint), device="cpu")
    agree = total = 0
    worst = worst_ref = 0.0
    for name, state, questions in CASES:
        items, _ = npu.prepare(state, questions)
        where = sorted({f"NPU {r[0]}x{r[1]}" if r else "CPU" for r in map(npu.route, items)})
        p_npu, p_cpu = _probs(npu, state, questions), _probs(cpu, state, questions)
        err = max(float(np.abs(p_npu[q] - p_cpu[q]).max()) for q in questions)
        same = sum(int(p_npu[q].argmax() == p_cpu[q].argmax()) if len(p_npu[q]) > 1
                   else int((p_npu[q][0] >= .5) == (p_cpu[q][0] >= .5)) for q in questions)
        line = f"{name:16s} {max(len(it['ids']) for it in items):4d} tokens  {', '.join(where):14s} " \
               f"same answer {same}/{len(questions)}  max |dp| NPU vs CPU {err:.1e}"
        if ref is not None:
            r = ref.predict(state, questions)["answers"]
            c = cpu.predict(state, questions)["answers"]
            e = max(abs(r[q].get("noul", 0) - c[q].get("noul", 0)) if c[q]["type"] == "noul" else
                    max(abs(r[q]["probabilities"][k] - v) for k, v in c[q]["probabilities"].items()) for q in questions)
            worst_ref = max(worst_ref, e)
            line += f"  |  CPU vs PyTorch {e:.1e}"
        print(line, flush=True)
        agree, total, worst = agree + same, total + len(questions), max(worst, err)
    print(f"\nNPU answers {agree}/{total} identical to the CPU model; max probability difference {worst:.1e}"
          + (f"; CPU model vs PyTorch max {worst_ref:.1e} (4-decimal rounding)" if ref is not None else ""))
    return 0 if agree == total else 1


def cmd_bench(a):
    from . import Agent
    from .examples import CASES, EMAIL, QUESTIONS

    one = {"department": QUESTIONS["department"]}
    long_name, long_state, long_q = CASES[-1]
    for device in a.device:
        agent = Agent(a.models, device=device)
        print(f"[{device}] loaded in {agent.load_s:.1f}s", flush=True)
        for label, state, qs in (("1 question, short email", EMAIL, one), ("3 questions, short email", EMAIL, QUESTIONS),
                                 (f"3 questions, {long_name}", long_state, long_q)):
            agent.predict(state, qs)
            agent.predict(state, qs)
            ts = []
            for _ in range(a.n):
                t = time.perf_counter()
                agent.predict(state, qs)
                ts.append(time.perf_counter() - t)
            ms = np.array(ts) * 1000
            print(f"[{device}] {label:30s} P50 {np.percentile(ms, 50):7.1f} ms  P95 {np.percentile(ms, 95):7.1f} ms", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m laya_snapdragon")
    p.add_argument("--models", default="models", help="model directory (default: ./models)")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="download, export and compile the NPU buckets")
    b.add_argument("--seq", type=int, nargs="+", default=[128, 256, 512], help="sequence-length buckets")
    b.add_argument("--markers", type=int, default=8, help="max options per question on the NPU")
    b.add_argument("--no-npu", action="store_true", help="only the CPU model")
    b.set_defaults(fn=cmd_build)

    r = sub.add_parser("predict", help="answer typed questions about a state")
    r.add_argument("--state", help="text or JSON; read from stdin if omitted")
    r.add_argument("--questions", required=True, help="JSON file: {id: {type, instructions, criteria}}")
    r.add_argument("--device", default="auto", choices=["auto", "npu", "cpu"])
    r.set_defaults(fn=cmd_predict)

    v = sub.add_parser("verify", help="check NPU answers against the CPU model (and PyTorch)")
    v.add_argument("--torch", action="store_true", help="also compare against upstream PyTorch Laya")
    v.set_defaults(fn=cmd_verify)

    n = sub.add_parser("bench", help="end-to-end latency")
    n.add_argument("--device", nargs="+", default=["auto", "cpu"], choices=["auto", "npu", "cpu"])
    n.add_argument("-n", type=int, default=20)
    n.set_defaults(fn=cmd_bench)

    a = p.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
