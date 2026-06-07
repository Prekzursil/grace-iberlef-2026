#!/usr/bin/env python3
"""Robust background orchestrator for the GRACE Track-1 Kaggle sweep.

Handles Kaggle API flakiness (transient "Notebook not found"/500/Conflict),
the 2-concurrent-GPU cap, queue stalls, and artifact collection.

Per job (a kernel dir with full T4 metadata + init.py):
  1. CPU "create" (init.py, no machine_shape/dataset) so the slug exists.
  2. Full "run" push (real script + T4 + dataset) when a GPU slot is free.
  3. Poll `kernels output`; completion = tee log has "DONE: model=" / "FAILED:".
  4. Copy artifacts to outputs_final/<slug>/ and record the dev macro F1.

Launch in the background; tail this script's stdout for live progress.
"""
from __future__ import annotations
import json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).parent
OUTF = ROOT / "outputs_final"; OUTF.mkdir(exist_ok=True)
USER = "prekzursil"
MAX_CONCURRENT = 2
POLL = 60

# jobs to launch (dirs). gt1-mrb-s42 already done (0.744) -> excluded.
TO_LAUNCH = [a for a in sys.argv[1:]] or [
    "gt1-bsc-s42", "gt1-mdb-s42", "gt1-mrb-s100",
    "gt1-bsc-s100", "gt1-rob-s100", "gt1-mdb-s100",
]
# already running outside the orchestrator -> just poll for completion
ALREADY = os.environ.get("ALREADY_RUNNING", "gt1-rob-s42").split() if os.environ.get("ALREADY_RUNNING", "gt1-rob-s42") else []


def sh(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)


def push(d: str) -> str:
    """Push kernel dir; classify outcome."""
    rc, out = sh(["kaggle", "kernels", "push", "-p", d])
    if "successfully pushed" in out:
        return "ok"
    if "session count of 2" in out or "Maximum batch GPU" in out:
        return "cap"
    if "Notebook not found" in out or "500 Server Error" in out or "Conflict" in out:
        return "transient"
    return "error:" + out.strip().splitlines()[-1][:80] if out.strip() else "error"


def create_then_run(slug: str) -> str:
    """2-step launch. Returns 'ok'|'cap'|'transient'|'error...'."""
    d = ROOT / slug
    meta = d / "kernel-metadata.json"
    full = json.loads(meta.read_text())
    # purge pycache
    for pc in d.rglob("__pycache__"):
        shutil.rmtree(pc, ignore_errors=True)
    # CPU create (mini)
    mini = {"id": full["id"], "title": full["title"], "code_file": "init.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": False}
    meta.write_text(json.dumps(mini, indent=2))
    for _ in range(3):
        if push(str(d)) in ("ok", "transient"):
            break
        time.sleep(5)
    meta.write_text(json.dumps(full, indent=2))  # restore full
    time.sleep(3)
    # full run push, brief retry on transient (outer loop rotates the queue)
    for _ in range(2):
        r = push(str(d))
        if r in ("ok", "cap"):
            return r
        time.sleep(6)
    return r


def collect(slug: str) -> dict:
    """Pull output; classify state. Terminal -> persist artifacts to outputs_final/."""
    d = OUTF / slug
    tmp = OUTF / (slug + "_tmp")
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    sh(["kaggle", "kernels", "output", f"{USER}/{slug}", "-p", str(tmp)])
    logs = list(tmp.glob("*_log.txt"))
    if not logs:
        return {"status": "queued"}
    txt = logs[0].read_text(encoding="utf-8", errors="replace")
    if "DONE: model=" in txt:
        shutil.rmtree(d, ignore_errors=True); tmp.rename(d)
        macro = None
        mj = list(d.glob("*_metrics.json"))
        if mj:
            try:
                macro = json.loads(mj[0].read_text())["best_model_dev"]["macro_f1"]
            except Exception:
                pass
        return {"status": "SUCCESS", "macro": macro}
    if "FAILED:" in txt:
        shutil.rmtree(d, ignore_errors=True); tmp.rename(d)
        fail = next((l for l in txt.splitlines() if l.startswith("FAILED:")), "FAILED")
        return {"status": "FAIL", "detail": fail[:120]}
    # running: surface latest epoch / heartbeat line
    epochs = [l.strip() for l in txt.splitlines() if l.strip().startswith("--- EPOCH")]
    hbs = [l.strip() for l in txt.splitlines() if "[hb] step=" in l]
    last = epochs[-1] if epochs else (hbs[-1] if hbs else "(model loading)")
    return {"status": "running", "last": last[:90]}


def main() -> None:
    pending = list(TO_LAUNCH)
    running = {s: time.time() for s in ALREADY}
    done: dict[str, dict] = {}
    failed: dict[str, dict] = {}
    attempts: dict[str, int] = {}
    print(f"[orch] start | launch={pending} | already_running={list(running)}", flush=True)
    for tick in range(600):  # up to 600*60s = 10h
        # 1) collect completions / surface live progress
        for slug in list(running):
            res = collect(slug)
            if res["status"] in ("SUCCESS", "FAIL"):
                done[slug] = res
                el = (time.time() - running.pop(slug)) / 60
                print(f"[orch] DONE {slug}: {res} ({el:.0f} min)", flush=True)
            else:
                print(f"[orch]   {slug}: {res['status']} | {res.get('last','')}", flush=True)
        # 2) fill free slots (ROTATE through pending so one wedged job can't block)
        i = 0
        while len(running) < MAX_CONCURRENT and i < len(pending):
            slug = pending[i]
            attempts[slug] = attempts.get(slug, 0) + 1
            print(f"[orch] launching {slug} (attempt {attempts[slug]}) ...", flush=True)
            r = create_then_run(slug)
            if r == "ok":
                running[slug] = time.time(); pending.pop(i)
                print(f"[orch] RUNNING {slug} ({len(running)} active)", flush=True)
            elif r == "cap":
                print("[orch] GPU cap hit; waiting for a slot", flush=True)
                break
            elif attempts[slug] >= 4:
                failed[slug] = {"status": "LAUNCH_FAIL", "detail": r}
                pending.pop(i)
                print(f"[orch] SKIP {slug} after {attempts[slug]} attempts ({r})", flush=True)
            else:
                print(f"[orch] {slug} -> {r}; rotating", flush=True)
                i += 1
        # 3) done?
        if not pending and not running:
            print("[orch] ALL JOBS COMPLETE", flush=True)
            break
        print(f"[orch] tick {tick}: running={list(running)} pending={pending} "
              f"done={ {k: v.get('macro', v['status']) for k, v in done.items()} } "
              f"failed={list(failed)}", flush=True)
        time.sleep(POLL)
    # final summary
    print("\n[orch] ===== FINAL =====", flush=True)
    for slug, res in {**done, **failed}.items():
        print(f"  {slug}: {res}", flush=True)


if __name__ == "__main__":
    main()
