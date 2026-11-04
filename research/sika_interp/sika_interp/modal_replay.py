"""Patched replays on Modal: vLLM with the patch hook (`steer_serve.SteerExtension`) and the
replay driver (`sika_interp.patched`) in one container, as `remote/pod.sh serve-steer` and
`steered` ran on a Prime pod. The server, the driver and its outputs all live in the
container; outputs are appended to the volume under replays/ and committed as they come.

    modal run -m sika_interp.modal_replay --model Qwen/Qwen3.5-9B --plan plan.json \\
        --config configs/qwen.toml --out patched-qwen-primary.jsonl
    modal run -m sika_interp.modal_replay --model Qwen/Qwen3.5-9B --plan plan.json \\
        --config configs/qwen.toml --out smoke.jsonl --limit 2 --samples 1   # smoke test

The driver runs in its own virtual environment with the sika-agent package, as on the pod,
so its dependencies never touch vLLM's. Traces are read from the volume (traces/<run>.jsonl,
put there with `modal volume put`), spec files from patch/specs/ (`modal_app.patch_specs`).
"""

import json
import os
import subprocess
import time
import tomllib
from pathlib import Path

import modal

MINISTRAL = "mistralai/Ministral-3-8B-Reasoning-2512"
ENV = Path(__file__).resolve().parent.parent.parent.parent / "environments" / "sika_agent"
VLLM_WHEEL = "https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
cache = modal.Volume.from_name("sika-interp-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    # The pod's install: vLLM's CUDA 12.9 wheel with the matching torch, and the prebuilt
    # kernels, which vLLM picks for the GPU it sees (so that step runs on one).
    .uv_pip_install(f"vllm @ {VLLM_WHEEL}", "mistral-common>=1.8.6", "ninja", "httpx", extra_options="--torch-backend=cu129")
    .run_commands("vllm download-kernels", gpu="H100")
    .env({"HF_HOME": "/cache/hf", "VLLM_SERVER_DEV_MODE": "1"})
    .add_local_dir(ENV, "/opt/sika_agent", copy=True, ignore=[".venv", "dist", "outputs", "**/__pycache__", "scripts/.cache"])
    # The driver's own environment, as the pod's client venv: the env package, the OpenAI client.
    .run_commands("python -m venv /opt/client && /opt/client/bin/pip install -q /opt/sika_agent openai httpx")
    .add_local_python_source("sika_interp")
)
app = modal.App("sika-interp-replay", image=image)

STEER = ["--worker-extension-cls", "sika_interp.steer_serve.SteerExtension", "--enforce-eager", "--optimization-level", "0",
         "--no-enable-prefix-caching", "--max-num-batched-tokens", "32768"]


def serve(model: str, gpu_share: float, logs: Path) -> tuple[list[subprocess.Popen], str, str]:
    """Start vLLM (and, for Ministral, the reasoning adapter in front of it) with the patch
    extension; returns the processes, the chat base URL and the server's own URL for RPCs."""
    common = ["--max-model-len", "32768", "--gpu-memory-utilization", str(gpu_share), "--enable-auto-tool-choice"]
    if model == MINISTRAL:
        cmd = ["vllm", "serve", model, "--port", "8001", *common, "--tokenizer_mode", "mistral", "--config_format", "mistral",
               "--load_format", "mistral", "--reasoning-parser", "mistral", "--tool-call-parser", "mistral", *STEER]
    else:
        cmd = ["vllm", "serve", model, "--port", "8000", *common, "--language-model-only", "--gdn-prefill-backend", "triton",
               "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder", *STEER]
    env = dict(os.environ, PYTHONPATH="/root")
    procs = [subprocess.Popen(cmd, stdout=(logs / "server.log").open("w"), stderr=subprocess.STDOUT, env=env)]
    if model == MINISTRAL:
        from huggingface_hub import hf_hub_download

        prompt = hf_hub_download(MINISTRAL, "SYSTEM_PROMPT.txt")
        procs.append(subprocess.Popen(
            ["python", "-m", "sika_interp.serving", "--upstream", "http://localhost:8001", "--port", "8002", "--reasoning-prompt", prompt],
            stdout=(logs / "adapter.log").open("w"), stderr=subprocess.STDOUT, env=env,
        ))
        return procs, "http://localhost:8002/v1", "http://localhost:8001"
    return procs, "http://localhost:8000/v1", "http://localhost:8000"


def wait_ready(url: str, procs: list[subprocess.Popen], logs: Path, minutes: int = 25) -> None:
    import httpx

    deadline = time.time() + 60 * minutes
    while time.time() < deadline:
        if procs[0].poll() is not None:
            raise RuntimeError("vLLM exited:\n" + tail(logs / "server.log"))
        try:
            if httpx.get(url.removesuffix("/v1") + "/health", timeout=5).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(5)
    raise RuntimeError("vLLM not ready in time:\n" + tail(logs / "server.log"))


def tail(path: Path, lines: int = 40) -> str:
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:]) if path.exists() else "(no log)"


@app.function(gpu="H200", volumes={"/cache": cache}, timeout=8 * 3600, cpu=8, memory=32768)
def replay(model: str, plan: dict, traces: list[str], sampling: dict, out: str, concurrency: int = 240, turns: int = 12,
           samples: int = 0, limit: int = 0, max_tokens: int = 4096, gpu_share: float = 0.85) -> dict:
    """Serve `model` with the patch hook and run the plan's replays, appending to
    replays/<out> on the volume (a rerun continues it). Returns the driver's exit code and
    the number of rows written; the server and driver logs sit beside the output."""
    started = time.time()
    logs = Path("/cache/replays/logs") / Path(out).stem
    logs.mkdir(parents=True, exist_ok=True)
    work = Path("/tmp/patched")
    work.mkdir(exist_ok=True)
    (work / "plan.json").write_text(json.dumps(plan))
    (work / "sampling.json").write_text(json.dumps(sampling))
    procs, base_url, rpc_url = serve(model, gpu_share, logs)
    try:
        wait_ready(base_url if model != MINISTRAL else "http://localhost:8001", procs, logs)
        if model == MINISTRAL:
            wait_ready(base_url, procs, logs, minutes=3)
        print(f"server ready after {time.time() - started:.0f}s", flush=True)
        cmd = ["/opt/client/bin/python", "-u", "-m", "sika_interp.patched", "--traces", *[f"/cache/traces/{t}" for t in traces],
               "--model", model, "--base-url", base_url, "--rpc", rpc_url, "--sampling", str(work / "sampling.json"),
               "--plan", str(work / "plan.json"), "--out", f"/cache/replays/{out}", "--concurrency", str(concurrency),
               "--turns", str(turns), "--samples", str(samples), "--limit", str(limit), "--max-tokens", str(max_tokens)]
        with (logs / "driver.log").open("w") as log:
            driver = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONPATH="/root"))
            seen = 0
            while driver.poll() is None:
                time.sleep(60)
                cache.commit()
                text = (logs / "driver.log").read_text(errors="replace")
                if len(text) > seen:
                    print(text[seen:], end="", flush=True)
                    seen = len(text)
            print((logs / "driver.log").read_text(errors="replace")[seen:], end="", flush=True)
    finally:
        for p in procs:
            p.terminate()
        cache.commit()
    rows = sum(1 for _ in open(f"/cache/replays/{out}")) if Path(f"/cache/replays/{out}").exists() else 0
    return {"exit": driver.returncode, "out": f"replays/{out}", "rows": rows, "seconds": round(time.time() - started),
            "server_log_tail": tail(logs / "server.log", 8) if driver.returncode else ""}


@app.local_entrypoint()
def main(model: str, plan: str, config: str, out: str, traces: str = "", concurrency: int = 240, turns: int = 12,
         samples: int = 0, limit: int = 0, gpu: str = "H200", gpu_share: float = 0.85):
    """`plan`: a replay plan JSON (`patching.replay_plan`); `config`: configs/<model>.toml for
    its sampling; `traces`: comma-separated names under traces/ on the volume (default: the
    three study runs of the model); `out`: the output name under replays/."""
    short = "ministral" if model == MINISTRAL else "qwen"
    names = traces.split(",") if traces else [f"study-{r}-{short}.jsonl" for r in ("extra-entw", "matched", "more-twins")]
    sampling = tomllib.loads(Path(config).read_text())["sampling"]
    result = replay.with_options(gpu=gpu).remote(model, json.loads(Path(plan).read_text()), names, sampling, out,
                                                 concurrency, turns, samples, limit, gpu_share=gpu_share)
    print(json.dumps(result, indent=1))
