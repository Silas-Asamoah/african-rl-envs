#!/usr/bin/env bash
# Serve Qwen3.5-9B and Ministral 3 8B Reasoning with vLLM on a Prime GPU pod and play
# sika-agent episodes against them. Run on the pod from a copy of the repo:
#
#   bash research/sika_interp/remote/pod.sh setup
#   bash research/sika_interp/remote/pod.sh serve qwen && bash .../pod.sh wait qwen
#   bash research/sika_interp/remote/pod.sh serve ministral && bash .../pod.sh wait ministral
#   bash research/sika_interp/remote/pod.sh eval qwen [vf-eval options, e.g. -n 60]
#   bash research/sika_interp/remote/pod.sh study qwen      # the study's 800 episodes
#   bash research/sika_interp/remote/pod.sh resume <run>    # re-run its missing rollouts
#
# Traces land in outputs/, server logs in logs/.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
HERE=$ROOT/research/sika_interp
VENVS=$HOME/venvs
LOGS=$ROOT/logs
export PATH="$VENVS/vllm/bin:$HOME/.local/bin:/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda
# Prime's H200 pods run a CUDA 12.4 driver, too old for vLLM's default CUDA 13 build.
VLLM_WHEEL=https://github.com/vllm-project/vllm/releases/download/v0.32.0/vllm-0.32.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl
MINISTRAL=mistralai/Ministral-3-8B-Reasoning-2512
# Share of GPU memory per server: 0.42 lets both models share one H200; one model alone can
# take about 0.85 and keep twice as many conversations in flight.
GPU_SHARE=${GPU_SHARE:-0.42}
mkdir -p "$LOGS"
log() { echo "[$(date +%T)] $*"; }
port() { case "$1" in qwen) echo 8000 ;; ministral) echo 8002 ;; *) echo "unknown model $1" >&2; exit 2 ;; esac; }

case "${1:-}" in
setup)
  command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  uv venv -q "$VENVS/vllm" --python 3.12
  uv pip install -q -p "$VENVS/vllm/bin/python" --torch-backend=cu129 "vllm @ $VLLM_WHEEL" "mistral-common>=1.8.6" ninja
  # Prebuilt FlashInfer kernels: the pod's CUDA toolkit can't build ones that match.
  vllm download-kernels
  uv venv -q "$VENVS/client" --python 3.12
  uv pip install -q -p "$VENVS/client/bin/python" "$ROOT/environments/sika_agent"
  "$VENVS/vllm/bin/python" -c "import vllm, torch, transformers; print('vllm', vllm.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"
  ;;
serve)
  case "${2:-}" in
  qwen)
    # Triton kernels for the linear-attention prefill, for the same reason.
    setsid nohup vllm serve Qwen/Qwen3.5-9B --port 8000 --max-model-len 32768 \
      --gpu-memory-utilization "$GPU_SHARE" --language-model-only --gdn-prefill-backend triton \
      --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
      > "$LOGS/qwen.log" 2>&1 < /dev/null &
    ;;
  ministral)
    setsid nohup vllm serve "$MINISTRAL" --port 8001 --max-model-len 32768 \
      --gpu-memory-utilization "$GPU_SHARE" --tokenizer_mode mistral --config_format mistral --load_format mistral \
      --reasoning-parser mistral --enable-auto-tool-choice --tool-call-parser mistral \
      > "$LOGS/ministral.log" 2>&1 < /dev/null &
    # Mistral's reasoning prompt goes in front of the server via sika_interp.serving (port 8002).
    prompt=$("$VENVS/vllm/bin/python" -c "from huggingface_hub import hf_hub_download as d; print(d('$MINISTRAL', 'SYSTEM_PROMPT.txt'))")
    (cd "$HERE" && setsid nohup "$VENVS/vllm/bin/python" -m sika_interp.serving \
      --upstream http://localhost:8001 --port 8002 --reasoning-prompt "$prompt" \
      > "$LOGS/ministral-proxy.log" 2>&1 < /dev/null &)
    ;;
  *) echo "serve qwen|ministral" >&2; exit 2 ;;
  esac
  log "$2 starting"
  ;;
wait)
  for name in "${@:2}"; do
    for _ in $(seq 1 80); do
      if curl -sf "localhost:$(port "$name")/v1/models" > /dev/null; then log "$name ready"; continue 2; fi
      if grep -q "Engine core initialization failed" "$LOGS/$name.log" 2> /dev/null; then
        log "$name failed: see $LOGS/$name.log"; exit 1
      fi
      sleep 15
    done
    log "$name not ready after 20 minutes"; exit 1
  done
  ;;
eval)
  name=${2:?eval qwen|ministral}
  shift 2
  logfile="$LOGS/eval-$name-$(date +%H%M%S)-$$.log"
  LOCAL_KEY=EMPTY "$VENVS/client/bin/vf-eval" sika-agent @ "$HERE/configs/$name.toml" \
    --no-push --no-rich -o "$ROOT/outputs" "$@" > "$logfile" 2>&1
  log "eval $name done, log $logfile"
  ;;
resume)
  # Re-run a run's missing or errored rollouts, e.g. after restarting a server.
  run=${2:?resume <run dir under outputs/>}
  (setsid nohup env LOCAL_KEY=EMPTY "$VENVS/client/bin/vf-eval" @ "$ROOT/outputs/$run/configs/resolved/eval.json" \
    --resume > "$LOGS/resume-$run-$(date +%H%M%S).log" 2>&1 < /dev/null &)
  log "resuming $run"
  ;;
study)
  # The study's episodes (sika_interp.tasks), in the background: the matched twins, and the
  # extra Twi–English seeds selected by task id.
  name=${2:?study qwen|ministral [matched|extra]}
  part=${3:-both}
  if [ "$part" != extra ]; then
    (setsid nohup bash "$0" eval "$name" --run.dir "study-matched-$name" < /dev/null > /dev/null 2>&1 &)
  fi
  if [ "$part" != matched ]; then
    ids=$(cd "$HERE" && "$VENVS/client/bin/python" -c \
      "import json; from sika_interp.tasks import extra_en_tw_tasks; print(json.dumps([t.data.id for t in extra_en_tw_tasks()]))")
    (setsid nohup bash "$0" eval "$name" --env.taskset.tasks_per_combo 20 --select.include.ids "$ids" \
      --run.dir "study-extra-entw-$name" < /dev/null > /dev/null 2>&1 &)
  fi
  log "study $name ($part) started"
  ;;
*)
  sed -n '2,10p' "$0"; exit 2
  ;;
esac
