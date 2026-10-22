"""
Run every consistency check and summarise.

    python run_all.py                 quick mode (representative parameters, ~7 minutes in total)
    python run_all.py --full          all parameter values of the paper's tables (~1.5 hours on one core; check_11
                                      takes ~40 min, check_07 ~30 min)
    python run_all.py 03 05           only the checks whose number is given
    python run_all.py --jobs 4        run up to 4 checks in parallel (with OMP_NUM_THREADS=1, so that numpy's BLAS
                                      does not oversubscribe the cores)

Each script writes its full output to results/<script>[_full].log; the summary lists the RESULT line of every script.
The exit code is 0 if every test passed.
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKS = sorted(f for f in os.listdir(os.path.join(HERE, "checks")) if f.startswith("check_") and f.endswith(".py"))


def run(script, full):
    log = os.path.join(HERE, "results", script.replace(".py", "_full.log" if full else ".log"))
    cmd = [sys.executable, os.path.join(HERE, "checks", script)] + (["--full"] if full else [])
    t0 = time.time()
    with open(log, "w") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=HERE)
    result = "no RESULT line (crashed?)"
    with open(log) as fh:
        for line in fh:
            if line.startswith("RESULT"):
                result = line.strip()
    return script, rc, result, time.time() - t0, log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("only", nargs="*", help="check numbers to run, e.g. 03 05 (default: all)")
    ap.add_argument("--full", action="store_true", help="full parameter sets (slow)")
    ap.add_argument("--jobs", type=int, default=1, help="number of checks run in parallel")
    A = ap.parse_args()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    todo = [c for c in CHECKS if not A.only or any(c.startswith(f"check_{n.zfill(2)}") for n in A.only)]
    print(f"running {len(todo)} checks ({'full' if A.full else 'quick'} mode, {A.jobs} in parallel); logs in results/", flush=True)
    failed = 0
    with ThreadPoolExecutor(max_workers=A.jobs) as ex:
        for script, rc, result, dt, log in ex.map(lambda c: run(c, A.full), todo):
            status = "PASS" if rc == 0 else "FAIL"
            failed += rc != 0
            print(f"[{status}] {script:36s} {dt:7.0f} s   {result}", flush=True)
    print(f"\n{len(todo) - failed} of {len(todo)} checks passed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
