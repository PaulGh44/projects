"""Shared helpers for the check scripts: import path, command-line options, and a uniform PASS/FAIL report.

Every check prints, for each test,
    label: value(s)   rel. diff  [ok]   or   [FAIL]
and ends with one summary line  "RESULT <script>: N tests, all passed"  (exit code 0) or "... k FAILED" (exit code 1).
Two kinds of tests are reported:
  * agree(): two independent computations of the same quantity (e.g. the two channels of a sewing constraint)
             agree to a relative tolerance;
  * paper(): a computed number reproduces the value printed in the paper, to the number of digits printed.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402


def options(doc, extra=None):
    """Standard options: --full runs the complete parameter set of the paper's tables (slow);
    the default ('quick') runs a representative subset in a few minutes."""
    ap = argparse.ArgumentParser(description=doc.strip().splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=doc)
    ap.add_argument('--full', action='store_true', help="all parameter values of the paper's tables (slow)")
    if extra:
        extra(ap)
    return ap.parse_args()


def fmt(z, d=12):
    z = complex(z)
    if abs(z.imag) <= 1e-9*max(abs(z.real), 1e-300):
        return f"{z.real:.{d}g}"
    if abs(z.real) <= 1e-9*abs(z.imag):
        return f"{z.imag:.{d}g}i"
    return f"{z.real:.{d}g}{z.imag:+.{d}g}i"


def _printed_tolerance(printed):
    """Half a unit in the last printed digit (relative), times a safety factor 3 for round-off."""
    s = printed.lower().replace('+', '')
    mant, _, ex = s.partition('e')
    ex = int(ex) if ex else 0
    mant = mant.lstrip('-')
    decimals = len(mant.split('.')[1]) if '.' in mant else 0
    return 3*0.5*10.0**(ex - decimals)


class Report:
    def __init__(self, script, title, reference):
        self.script = os.path.basename(script)
        self.n = 0
        self.failed = []
        self.t0 = time.time()
        bar = "=" * 100
        print(bar)
        print(f"{self.script}: {title}")
        print(f"paper: {reference}")
        print(bar, flush=True)

    def section(self, text):
        print(f"\n--- {text}", flush=True)

    def note(self, text):
        print(f"    {text}", flush=True)

    def _record(self, label, ok):
        self.n += 1
        if not ok:
            self.failed.append(label)
        return "ok" if ok else "FAIL"

    def agree(self, label, a, b, tol, names=("lhs", "rhs")):
        """Relative agreement |a - b|/|a| <= tol (for arrays: the largest relative difference)."""
        a, b = np.broadcast_arrays(np.asarray(a, dtype=complex), np.asarray(b, dtype=complex))
        if a.ndim:
            r = float(np.max(np.abs(a - b)/np.abs(a)))
            st = self._record(label, r <= tol)
            print(f"  {label}: {a.size} values, e.g. {names[0]} = {fmt(a.flat[0])}   {names[1]} = {fmt(b.flat[0])}   "
                  f"max rel. diff {r:.1e}   [{st}, tol {tol:.0e}]", flush=True)
            return r
        r = float(abs(a - b)/abs(a))
        st = self._record(label, r <= tol)
        print(f"  {label}: {names[0]} = {fmt(a)}   {names[1]} = {fmt(b)}   rel. diff {r:.1e}   [{st}, tol {tol:.0e}]",
              flush=True)
        return r

    def ratio(self, label, a, b, expected, tol):
        """a/b equals `expected` to relative tolerance tol."""
        q = a/b
        r = abs(q/expected - 1)
        st = self._record(label, r <= tol)
        print(f"  {label}: ratio = {fmt(q)}   expected {fmt(expected)}   rel. diff {r:.1e}   [{st}, tol {tol:.0e}]",
              flush=True)
        return q

    def small(self, label, x, tol):
        """|x| <= tol."""
        st = self._record(label, abs(x) <= tol)
        print(f"  {label}: {abs(x):.1e}   [{st}, tol {tol:.0e}]", flush=True)

    def true(self, label, cond, detail=""):
        st = self._record(label, bool(cond))
        print(f"  {label}: {detail}   [{st}]", flush=True)

    def paper(self, label, value, printed, part='real', slack=1.0, rtol=None):
        """Compare with a number printed in the paper (string, e.g. '7.5621654080' or '2.1076975609e-2'),
        to half a unit of its last printed digit (x3 for round-off, x slack), or to the relative tolerance rtol
        when the printed digits exceed the accuracy of the computation that produced them."""
        v = complex(value)
        x = {'real': v.real, 'imag': v.imag, 'abs': abs(v)}[part]
        ref = float(printed)
        tol = _printed_tolerance(printed)*slack if rtol is None else rtol*abs(ref)
        ok = abs(x - ref) <= tol
        st = self._record(label, ok)
        print(f"  {label}: computed {x:.12g}   paper {printed}   |diff| {abs(x - ref):.1e}   [{st}]", flush=True)

    def done(self):
        dt = time.time() - self.t0
        print()
        if self.failed:
            print(f"RESULT {self.script}: {self.n} tests, {len(self.failed)} FAILED: {', '.join(self.failed)}   [{dt:.0f} s]")
            sys.exit(1)
        print(f"RESULT {self.script}: {self.n} tests, all passed   [{dt:.0f} s]")
        sys.exit(0)
