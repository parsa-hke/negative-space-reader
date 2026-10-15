"""Lab session driver for the UOWD Windows GPU machine (plan Appendix D; decision record 002).

Run from the repository root with the lab virtual environment's Python:

    python lab_session.py check                 environment, GPUs, llama.cpp, model files, data
    python lab_session.py serve                 one llama-server per GPU (Qwen2.5-32B Q4_K_M), waits until healthy
    python lab_session.py status                health of every server
    python lab_session.py eye                   task 3.1: eye test on the 32B, next to the local 7B run
    python lab_session.py extract --genre G     task 3.2: claims for 500 dev documents of genre G (privacy | papers);
                                                run both genres at once in two windows: they share the servers
    python lab_session.py pool                  optional: claims for 100 train papers (presence-label pool, plan §4.1)
    python lab_session.py stop                  stop the llama-servers (frees the GPUs for embedding)
    python lab_session.py embed [--model 8b]    section and title embeddings (0.6B, or the 8B ablation at 1024 dims)
    python lab_session.py pack                  everything to take home -> lab_out/take_home.zip

Paths default to C:\\negspace_lab\\llama and C:\\negspace_lab\\models; override with --llama-dir / --model-file.
Every job is a normal experiments.run call, so it is logged in runs/log.csv; a job cut off by the end of the session
is rerun with the same command and replays finished teacher calls from the cache.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
OUT = REPO / "lab_out"
SERVERS = OUT / "servers.json"
LAB_ROOT = Path(os.environ.get("NEGSPACE_LAB_ROOT", r"C:\negspace_lab"))
DEFAULT_LLAMA = LAB_ROOT / "llama"
DEFAULT_MODEL = LAB_ROOT / "models" / "qwen2.5-32b" / "qwen2.5-32b-instruct-q4_k_m-00001-of-00005.gguf"
PORTS = [8081, 8082, 8083, 8084]


def say(msg: str) -> None:
    print(f"[lab] {msg}", flush=True)


def run_experiment(args: list[str]) -> int:
    env = {**os.environ, "PYTHONUTF8": "1"}
    cmd = [sys.executable, "-m", "experiments.run", *args]
    say(" ".join(cmd))
    return subprocess.call(cmd, cwd=REPO, env=env)


def server_exe(llama_dir: Path) -> Path:
    for name in ("llama-server.exe", "llama-server"):
        hits = list(llama_dir.rglob(name))
        if hits:
            return hits[0]
    raise SystemExit(f"llama-server not found under {llama_dir} (unzip the llama.cpp and cudart zips there)")


def healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
            return json.loads(r.read().decode()).get("status") == "ok"
    except Exception:
        return False


# ------------------------------------------------------------------------------------ commands
def cmd_check(a: argparse.Namespace) -> None:
    OUT.mkdir(exist_ok=True)
    ok = True
    say(f"python {platform.python_version()} at {sys.executable}")
    if not platform.python_version().startswith("3.11"):
        say("WARNING: expected Python 3.11")
    try:
        import torch

        n = torch.cuda.device_count()
        say(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, GPUs: {n}")
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            say(f"  cuda:{i} {p.name} {p.total_memory / 2**30:.1f} GiB")
        ok &= n >= 1
    except Exception as e:  # noqa: BLE001
        say(f"torch problem: {e}")
        ok = False
    try:
        say(f"llama-server: {server_exe(a.llama_dir)}")
    except SystemExit as e:
        say(str(e))
        ok = False
    parts = sorted(a.model_file.parent.glob("qwen2.5-32b-instruct-q4_k_m-*.gguf"))
    size = sum(p.stat().st_size for p in parts) / 1e9
    say(f"32B model files: {len(parts)}/5 in {a.model_file.parent} ({size:.1f} GB of 19.9)")
    ok &= len(parts) == 5 and size > 19.5
    for rel in ("data/processed/privacy/sections.jsonl", "data/processed/papers/sections.jsonl",
                "data/processed/splits/privacy.json", "prompts/eval/induce_extract_eye_test.json",
                "runs/eye_v2_s0/eye_test_privacy.jsonl"):
        exists = (REPO / rel).is_file()
        ok &= exists
        say(f"{'ok     ' if exists else 'MISSING'} {rel}")
    with open(OUT / "device_info.txt", "w", encoding="utf-8") as f:
        subprocess.call([sys.executable, "-m", "experiments.run", "--device-info"], cwd=REPO, stdout=f,
                        env={**os.environ, "PYTHONUTF8": "1"})
    say(f"device report written to {OUT / 'device_info.txt'}")
    say("ALL GOOD" if ok else "NOT READY: fix the lines above")


def cmd_serve(a: argparse.Namespace) -> None:
    import psutil

    OUT.mkdir(exist_ok=True)
    exe = server_exe(a.llama_dir)
    if not a.model_file.is_file():
        raise SystemExit(f"model file missing: {a.model_file}")
    gpus = [int(g) for g in a.gpus.split(",")]
    procs = []
    flags = 0
    if os.name == "nt":  # detached: closing this window does not stop the servers
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    for gpu, port in zip(gpus, PORTS):
        if healthy(port):
            say(f"port {port} already serving; skipped")
            continue
        log = open(OUT / f"llama_{port}.log", "w", encoding="utf-8")
        cmd = [str(exe), "-m", str(a.model_file), "-ngl", "all", "-c", str(a.ctx), "-np", str(a.parallel),
               "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--host", "127.0.0.1", "--port", str(port)]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, creationflags=flags)
        procs.append({"pid": p.pid, "port": port, "gpu": gpu})
        say(f"started llama-server on GPU {gpu}, port {port} (pid {p.pid}), log {OUT / f'llama_{port}.log'}")
    old = json.loads(SERVERS.read_text()) if SERVERS.exists() else []
    SERVERS.write_text(json.dumps([s for s in old if psutil.pid_exists(s["pid"])] + procs, indent=1))
    t0 = time.time()
    while time.time() - t0 < a.wait:
        up = [p for p in PORTS[:len(gpus)] if healthy(p)]
        if len(up) == len(gpus):
            say(f"all {len(gpus)} servers healthy after {time.time() - t0:.0f}s")
            return
        dead = [s for s in procs if not psutil.pid_exists(s["pid"])]
        if dead:
            raise SystemExit(f"server(s) exited: {dead}; read the log files in {OUT}")
        time.sleep(5)
    raise SystemExit(f"servers not healthy after {a.wait}s; check {OUT}/llama_*.log")


def cmd_status(a: argparse.Namespace) -> None:
    for port in PORTS:
        say(f"port {port}: {'healthy' if healthy(port) else 'down'}")


def cmd_stop(a: argparse.Namespace) -> None:
    import psutil

    for s in (json.loads(SERVERS.read_text()) if SERVERS.exists() else []):
        if psutil.pid_exists(s["pid"]):
            psutil.Process(s["pid"]).kill()
            say(f"stopped pid {s['pid']} (port {s['port']})")
    SERVERS.unlink(missing_ok=True)


def cmd_eye(a: argparse.Namespace) -> None:
    sys.exit(run_experiment(["--config", "configs/induce.yaml", "--overlay", "lab", "--name", "eye_v2_32b",
                             "--set", "induce.eye_test.compare_with=runs/eye_v2_s0"]))


def cmd_extract(a: argparse.Namespace) -> None:
    if a.genre not in ("privacy", "papers"):
        raise SystemExit("choose --genre privacy or --genre papers (run both, in two windows)")
    sys.exit(run_experiment(["--config", "configs/induce.yaml", "--overlay", "lab", "--name", f"extract_32b_{a.genre}",
                             "--set", f"pack={a.genre}", "--set", "induce.stage=extract",
                             "--set", f"induce.extract.documents_per_genre={a.docs}"]))


def cmd_pool(a: argparse.Namespace) -> None:
    sys.exit(run_experiment(["--config", "configs/induce.yaml", "--overlay", "lab", "--name", "extract_32b_trainpool",
                             "--set", "pack=papers", "--set", "induce.stage=extract",
                             "--set", "induce.extract.split=train", "--set", "induce.extract.documents_per_genre=100"]))


def cmd_embed(a: argparse.Namespace) -> None:
    args = ["--config", "configs/embed.yaml", "--overlay", "lab"]
    if a.model == "8b":
        args += ["--name", "embed_8b", "--set", "models.embedder.id=Qwen/Qwen3-Embedding-8B",
                 "--set", "models.embedder.truncate_dim=1024", "--set", "models.embedder.batch_size=16"]
    else:
        args += ["--name", "embed_0.6b"]
    sys.exit(run_experiment(args))


def cmd_pack(a: argparse.Namespace) -> None:
    OUT.mkdir(exist_ok=True)
    target = OUT / "take_home.zip"
    picks = [REPO / "runs" / "log.csv", *OUT.glob("*.txt"), *OUT.glob("*.log"), *OUT.glob("*.json")]
    for pattern in ("eye_v2_32b_s0*", "extract_32b*", "embed_*", "vllm_check*"):
        for d in (REPO / "runs").glob(pattern):
            picks += [p for p in d.rglob("*") if p.is_file()]
    for root in (REPO / ".cache" / "negspace" / "teacher", REPO / "data" / "embeddings"):
        if root.exists():
            picks += [p for p in root.rglob("*") if p.is_file()]
    with zipfile.ZipFile(target, "w") as z:
        for p in picks:
            method = zipfile.ZIP_STORED if p.suffix == ".npy" else zipfile.ZIP_DEFLATED
            z.write(p, p.relative_to(REPO).as_posix(), compress_type=method)
    say(f"{target} ({target.stat().st_size / 1e9:.2f} GB, {len(picks)} files). Unzip it in the repository root at home.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "serve", "status", "eye", "extract", "pool", "stop", "embed", "pack"])
    ap.add_argument("--llama-dir", type=Path, default=DEFAULT_LLAMA)
    ap.add_argument("--model-file", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--parallel", type=int, default=8, help="slots per server")
    ap.add_argument("--ctx", type=int, default=49152, help="total context per server (slots x 6144)")
    ap.add_argument("--wait", type=int, default=900, help="seconds to wait for servers to load")
    ap.add_argument("--docs", type=int, default=500)
    ap.add_argument("--genre", default=None)
    ap.add_argument("--model", choices=["0.6b", "8b"], default="0.6b")
    a = ap.parse_args()
    {"check": cmd_check, "serve": cmd_serve, "status": cmd_status, "eye": cmd_eye, "extract": cmd_extract,
     "pool": cmd_pool, "stop": cmd_stop, "embed": cmd_embed, "pack": cmd_pack}[a.command](a)


if __name__ == "__main__":
    main()
