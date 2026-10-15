#!/usr/bin/env bash
# Lab session driver (plan Appendix D). Run the steps in order from the repository root:
#
#   ./lab_session.sh setup        venv + vLLM, GPU report, teacher choice           (~10 min)
#   ./lab_session.sh download     teacher weights into $HF_HOME                     (depends on the network)
#   ./lab_session.sh vllm-check   small model through vLLM, 20 sections              (~5 min)
#   ./lab_session.sh eye          task 3.1: 72B eye test next to the local 7B run     (~10 min incl. loading)
#   ./lab_session.sh extract      task 3.2: 72B extraction, 500 dev documents/genre  (~1-3 h, resumable)
#   ./lab_session.sh pack         everything to take home -> lab_out/take_home.tar.gz
#
# Every run is logged in runs/log.csv. If a step is cut off, run it again: finished teacher calls come back from
# the response cache in seconds. Set HF_HOME to a large disk before `download` (the 72B AWQ build is 41.6 GB).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p lab_out
PY=.venv/bin/python
ARGS_FILE=lab_out/teacher.args

teacher_args() { [ -f "$ARGS_FILE" ] && cat "$ARGS_FILE" || { echo "run ./lab_session.sh setup first" >&2; exit 1; }; }

case "${1:-}" in
setup)
  if [ ! -x "$PY" ]; then
    if command -v python3.11 >/dev/null; then python3.11 -m venv .venv
    elif command -v uv >/dev/null; then uv venv --python 3.11 .venv
    else echo "Need Python 3.11 (python3.11, or install uv: curl -LsSf https://astral.sh/uv/install.sh | sh)"; exit 1; fi
  fi
  $PY -m pip install -q --upgrade pip
  $PY -m pip install -q -e ".[gpu]"
  $PY -m experiments.run --device-info | tee lab_out/device_info.txt
  $PY -m experiments.run --device-info --json > lab_out/device_info.json
  # Teacher build from the VRAM, plan §0.2 #6 / §0.6. FORCE_AWQ=1 prefers the 41.6 GB AWQ build over the 145 GB
  # bf16 one when both fit (faster download in a day session).
  $PY - <<'PYEOF'
import json, os
info = json.load(open("lab_out/device_info.json"))
gpus = [d for d in info["devices"] if d["type"] == "cuda"]
n, per = len(gpus), min((d["vram_gib"] for d in gpus), default=0)
force_awq = os.environ.get("FORCE_AWQ") == "1"
# bf16 weights are 145 GB: they need 4 x 80 GB (2 x 80 GB leaves no room for the KV cache).
if n >= 4 and per >= 75 and not force_awq:
    args, why = ["models.teacher.id=Qwen/Qwen2.5-72B-Instruct", "models.teacher.quantization=null",
                 "models.teacher.dtype=bfloat16", "models.teacher.tensor_parallel_size=4"], f"{n}x{per} GiB: 72B bf16, TP=4"
elif per >= 75 or (n >= 2 and per >= 44):
    tp = 4 if n >= 4 else 2 if n >= 2 else 1  # AWQ on every GPU available: more KV cache, more throughput
    args, why = [f"models.teacher.tensor_parallel_size={tp}"], f"{n}x{per} GiB: 72B AWQ, TP={tp}"
elif per >= 22:
    args, why = ["models.teacher.id=Qwen/Qwen2.5-32B-Instruct-AWQ", "models.teacher.tensor_parallel_size=1",
                 "models.teacher.max_model_len=6144"], f"{n}x{per} GiB: too small for the 72B, 32B AWQ fallback (plan §0.2 #6)"
else:
    raise SystemExit(f"{n} GPU(s) with {per} GiB: no usable teacher on this machine; use --overlay openrouter")
open("lab_out/teacher.args", "w").write(" ".join(f"--set {a}" for a in args))
model = next((a.split("=", 1)[1] for a in args if a.startswith("models.teacher.id=")), "Qwen/Qwen2.5-72B-Instruct-AWQ")
open("lab_out/teacher.model", "w").write(model)
print(f"\nTeacher: {why}\nModel to download: {model}")
PYEOF
  echo "Next: export HF_HOME=<large disk>/hf ; ./lab_session.sh download"
  ;;
download)
  model=$(cat lab_out/teacher.model)
  echo "Downloading $model into ${HF_HOME:-~/.cache/huggingface}"
  .venv/bin/hf download "$model"
  .venv/bin/hf download Qwen/Qwen2.5-1.5B-Instruct
  ;;
vllm-check)
  $PY -m experiments.run --config configs/induce.yaml --overlay gpu --name vllm_check --set pack=privacy \
    --set models.teacher.id=Qwen/Qwen2.5-1.5B-Instruct --set models.teacher.quantization=null \
    --set models.teacher.dtype=bfloat16 --set models.teacher.tensor_parallel_size=1
  echo "Pass = [ok] above and runs/vllm_check_s0/eye_test.html shows 20 privacy sections with claims."
  ;;
eye)
  # shellcheck disable=SC2046
  $PY -m experiments.run --config configs/induce.yaml --overlay gpu --name eye_v2_72b \
    --set induce.eye_test.compare_with=runs/eye_v2_s0 $(teacher_args)
  echo "Review runs/eye_v2_72b_s0/eye_test.html against the pass criteria (plan Appendix D)."
  ;;
extract)
  # shellcheck disable=SC2046
  $PY -m experiments.run --config configs/induce.yaml --overlay gpu --name extract_72b \
    --set induce.stage=extract $(teacher_args)
  ;;
pack)
  tar -czf lab_out/take_home.tar.gz lab_out/device_info.* lab_out/teacher.* runs/log.csv \
    $(ls -d runs/vllm_check_s0* runs/eye_*72b_s0* runs/extract_72b_s0* 2>/dev/null) \
    $( [ -d .cache/negspace/teacher ] && echo .cache/negspace/teacher )
  ls -la lab_out/take_home.tar.gz
  echo "Copy lab_out/take_home.tar.gz home and run: tar -xzf take_home.tar.gz (in the repository root)."
  ;;
*)
  sed -n '2,13p' "$0"; exit 1 ;;
esac
