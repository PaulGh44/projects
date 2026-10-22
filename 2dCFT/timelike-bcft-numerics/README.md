# Numerical checks for "Liouville boundary conformal field theory at c ≤ 1"

Tsiares, Betzios, Papadoulaki, Ghiringhelli — code accompanying the Letter and its Supplemental Material
(`main_LiouvilleBCFT.tex`).

This package defines the boundary data of Liouville theory at c ≤ 1 (Table I of the Letter) in **one** place,
`tlbcft/theory.py`, and verifies every consistency condition reported in Table II and in the Supplemental Material,
one script per condition. Each script states what it tests, with the formulas and the paper's equation numbers,
prints both sides of every comparison, compares with the numbers printed in the paper, and ends with a single
`RESULT ... all passed` line.

It is pure Python (numpy + scipy). Nothing needs to be compiled.

---

## 1. Installation and quick start

Python ≥ 3.9 with numpy ≥ 1.22 and scipy ≥ 1.8 (no other dependencies). The reference logs in `reference_logs/` were produced
with Python 3.13, numpy 2.5 and scipy 1.18 on Linux with OMP_NUM_THREADS=1.

```bash
cd timelike-bcft-numerics
python -m venv .venv && source .venv/bin/activate        # optional
pip install -r requirements.txt                          # numpy, scipy

python run_all.py                  # all 16 checks, quick mode: ~7 min
OMP_NUM_THREADS=1 python run_all.py --jobs 4   # the same, four checks in parallel
python run_all.py --full           # every parameter value of the paper's tables: ~1.5 h
python checks/check_05_disc_two_point.py          # a single check (add --full for all central charges)
python -m tlbcft.kernels                          # self-test of one library module
```

`run_all.py` writes one log per check to `results/` (created on first run). `reference_logs/` holds the logs of the
runs made when the package was assembled: quick mode for every check (`check_XX_*.log`), and full mode for every check
that has one (`check_XX_*_full.log`), so you can compare your output line by line, e.g.
`diff <(grep -v '\[.*s\]$' results/check_04_cylinder.log) <(grep -v '\[.*s\]$' reference_logs/check_04_cylinder.log)`
(the filter drops the timing lines).

---

## 2. Layout

```
tlbcft/                  the library (definitions; no physics claims are tested here)
  special.py             Barnes G; Barnes double Gamma Gamma_b and S_b: exact at rational b^2, integral rep. at any b
  kernels.py             c >= 25 fusion kernel F (closed form at b^2 = m/n; Teschner-Vartanov integral at any b),
                         modular kernel S (from F at b' = sqrt2 b), and the Virasoro-Wick rotations RF, RS
  blocks.py              sphere four-point blocks (elliptic form, any c), torus one-point blocks
  comb5.py               five-point comb blocks for the disc with two bulk fields and one boundary field
  theory.py              Table I: bulk and boundary data at c <= 1 (class Timelike) and c >= 25 (class Spacelike)
  numerics.py            contour grids on eps + iR, Simpson/midpoint rules, circle residues
checks/
  _common.py             command-line options and the PASS/FAIL report
  check_00 ... check_15  one script per consistency condition (table below)
run_all.py               runs all checks, writes results/*.log, prints a summary
reference_logs/          the logs of the reference runs (quick and full)
```

---

## 3. Conventions needed to read the code

**Parameters.** c ≥ 25: `b > 0`, `Q = b + 1/b`, `h = Q²/4 − P²`, spectrum `P ∈ iℝ`.
c ≤ 1: `b = i·bhat`, `chat = 1 − 6(bhat − 1/bhat)²`, `hhat(P) = (chat − 1)/24 − P²`, spectrum `P ∈ ε + iℝ`.
A rational `bhat² = m/n` is passed as a tuple, `Timelike((1, 2))` (closed-form kernels, fast, vectorised);
any other value as a float, `Timelike(2**-0.5)` (Barnes integrals, slow, ~1 s per kernel).

| chat | 1 | −2 | 0 | −22/5 | −7 | 11/35 | 0.272 | −5 |
|---|---|---|---|---|---|---|---|---|
| bhat² | 1/1 | 1/2 | 2/3 | 2/5 | 1/3 | 5/7 | 1/√2 | (3−√5)/2 |

**Argument order of the kernels (important).**
`F(Ps, Pt, P1, P2, P3, P4)` means **F_{Ps,Pt}[P2 P3; P1 P4]**: operators P1, P2, P3, P4 at 0, z, 1, ∞; the s-channel fuses
(P1,P2), the t-channel (P2,P3). The helper `Fm(Ps, Pt, A, B, C, D)` is the paper's matrix order **F_{Ps,Pt}[A B; C D]**.
`RF`, `RFm` likewise. `S(Ps, Pt, P0)` = S_{Ps,Pt}[P0].

**Virasoro–Wick rotation (paper Eq. (2)).**
`RF_{Ps,Pt}[P2 P3; P1 P4] = (Pt/Ps) F_{iPt,iPs}[iP2 iP1; iP3 iP4]`, `RS_{Ps,Pt}[P0] = (Pt/Ps) S_{iPt,iPs}[iP0]`,
with F, S the c ≥ 25 kernels at b = bhat (`kernels.FusionKernel.RF`, `kernels.ModularKernel.RS`).

**Table I in the code** (`tlbcft/theory.py`, class `Timelike`, `normalization='final'`):

| paper | code | formula |
|---|---|---|
| ρ̂(P) | `rhoh(P)` | P²/(2 ρ0(iP)),  ρ0(P) = −4√2 sin(2πbP) sin(2πP/b) |
| Ĉ(P1,P2,P3) | `Ch(P1,P2,P3)` | 1/C0(iP1,iP2,iP3), C0 = DOZZ, Eq. (S3) |
| Ŝ(P;σ1,σ2) | `Sh(P,s1,s2)` | 2√2 ρ0(iσ1) ρ0(iσ2) C0(iP,iσ1,iσ2)/(σ1 σ2 P) |
| (Ĉ)^{σ3σ2σ1}_{P3P2P1} | `Ch3(P3,P2,P1,s3,s2,s1)` | 2 Ŝ(P3;σ1,σ3) RF_{σ2,P3}[P2 P1; σ3 σ1] |
| R̂(Pα,Pβ\|σ) | `Rh2(Pa,Pb,s)` | 2√2 RS_{σ,Pα}[Pβ]/Pα |
| R̂(Pα\|σ) | `Rh1(Pa,s)` | 4 cos(4πσPα)/Pα |
| Ψ0 coefficient R̂(Pα,0̂\|σ) | `Rh_psi0(Pa,s)` | 4 sin(4πσPα)/Pα |

`normalization='literal'` gives the literal rotation (k = 1), and `scales=(k1,k2,k3,k4)` any member of the family of
Sec. S3. The c ≥ 25 data of the middle column are `Spacelike`: `S0`, `C3`, `R2`, `R1`.

**Sewing.** Closed channels use the measure `rhoh(P) dP/i`, open channels `dP/(i Sh)`, all on `P = ε + iy`
(`dP/i = dy`). The amplitudes of Eq. (4) are written out literally in the check scripts (e.g. `four_point` in
check_03, `cylinder` in check_04, `disc_two_point` in check_05).

---

## 4. The checks

| script | paper | what is verified | quick | full |
|---|---|---|---|---|
| `check_00_numerics` | Secs. S1, S8 | Γ_b (two independent methods), F closed form vs Teschner–Vartanov integral, S from F vs Teschner's integral, general-b kernel vs closed form, blocks vs exact c = 1 block, torus vs sphere blocks (Poghossian) | 20 s | — |
| `check_01_table_I` | Table I; Eqs. (S6), (S8)–(S12), (S19)–(S20) | Ĉ solves the shift equation; vacuum rows; special columns of S, RS; tetrahedral/exchange identity; cyclicity and parities of the vertex; weight-one pole of Ŝ (−98.72744079/δ); bulk–boundary columns, Ψ0 as a label derivative | 1 s | — |
| `check_02_real_line_identities` | Table II "real-line identities"; Secs. S3, S6 | rotated pentagon; rotated disc identity (holds iff κ = 1/√2; values 3.308295403120, 10.980925470946); transported VΨΨ and VVΨ identities with the h → 1 − ĥ phases (and the −1, −i with h → ĥ) | 1.5 min | 2 min |
| `check_03_boundary_crossing` | Eq. (4); Table S-4pt | four-point crossing on ε + iℝ | 15 s | 1 min |
| `check_04_cylinder` | Eq. (S15); Table S-cyl | cylinder with a boundary insertion, open = closed | 10 s | 15 s |
| `check_05_disc_two_point` | Eq. (4); Table S-disc2 | disc two-point function, bulk = boundary; κ × 4 ⇒ ratio 4 | 20 s | 10 min |
| `check_06_disc_VPsiPsi` | Eqs. (S26)–(S28); Table S-vpp | disc VΨΨ, channel (i) = (ii) | 2 s | 8 min |
| `check_07_disc_VVPsi` | Eq. (4); Sec. S6; Table S-vvp | disc VVΨ, bulk = boundary; literal rotation gives bulk = 2 × boundary ⇒ k² = 2 | 20 s | 30 min |
| `check_08_annulus` | Eq. (4); Eqs. (S21)–(S22) | annulus = positive integer sum of Verma characters on the label-shifted lattice; quantized examples; Bautista–Bawane and Giribet–Sivilotti annuli | 10 s | — |
| `check_09_psi0_cylinder` | Eq. (S23) | Ψ0 trace kernel (closed form = integral = lattice sum), residues m_rs e_a/(4πi), modular covariance for rational and irrational bhat | 20 s | — |
| `check_10_kac_cancellation` | Eq. (S13); Table S-kac | T1 + T2 = 0 in the s-, t-channels, cylinder and disc; ingredients (a)–(c) of the proof; no cancellation in the bulk channel | 30 s | 2 min |
| `check_11_irrational_b` | Sec. S8; Tables S-4pt, S-cyl, S-kac | bhat² = 1/√2 and the golden-ratio point: Kac cancellation, Ψ0 residues; (full) crossing and cylinder | 3 min | 40 min |
| `check_12_c25_baseline` | Sec. S2; Eqs. (S28), (S31) | c ≥ 25: balancing, Verlinde, disc, pentagon, c = 25 theta identity, S0 > 0 and its vacuum pole, ZZ annulus, c = 28 amplitudes; (full) the two c = 28 identities behind the three-insertion discs | 10 s | 1.5 min |
| `check_13_half_zz` | Sec. S4 "Half-ZZ branes"; Table S-hzz | annuli of half-ZZ branes with our states (multiplicity one iff k² = 2), Ψ_(m,±) from Ψ0, negative self-annuli of ZZ-type differences, the half-ZZ disc fit (N1, c_(1,1), c_(1,3), factorization; full: ZZ_(2,+), N1/N2 = −2cos πbhat²) | 35 s | 2 min |
| `check_14_normalization` | Sec. S3 | the free constants: scaling laws of every density, the invariant (S17), Table I / literal factors, the odd-pairing form (S18) | 1 s | — |
| `check_15_physical_kernels` | Sec. S7 | parity (odd ⇒ residue sum, even ⇒ principal value); c = 1 family: VWR weights self-dual, fixed-sheet weights not; ĉ = −2: the physical kernels miss crossing by 8–48 % and depend on ε | 15 s | — |

Times: one core of a 2-core cloud machine; a recent laptop is typically faster. The full check_07 is dominated by the
level-12 comb blocks and the 11/35 rows. check_05, check_06 and check_07 at bhat² = 5/7 are slow because the closed-form
modular kernel needs the fusion kernel at b'² = 10/7 (mn = 70). The full check_11 evaluates about 4800 Barnes-integral
kernels (~0.5 s each).

---

## 5. Accuracy, and differences from the numbers printed in the paper

All checks reproduce the paper's printed values to the printed digits, except where noted below. Most of the
differences come from one change in the numerics (the first item); the others are accuracy limits of particular
kernels or grids.

* **Gram matrices are solved, not inverted.** The blocks need Gram matrices up to level 12, whose entries grow like hᴺ.
  The original `vblock.py` inverted them; `tlbcft.blocks` solves the scaled linear system instead. The difference
  matters where the blocks are evaluated at large nome, τ = ±½ + i/2 (|q| = e^{−π/2}) for the disc VΨΨ: against the
  exact c = 1, h_i = 1/16 block at level 12, the explicit inverse is off by 5·10⁻⁵–3·10⁻³ for |P| = 0.6–2, the solve
  by 4·10⁻¹⁰–6·10⁻⁸ (check_00, section 6, prints both). At ĉ = −2 the original code's blocks at τ = ½ + i/2 differ
  from the new ones by 2·10⁻⁵–5·10⁻⁴, while the new ones change by only 5·10⁻¹⁰–5·10⁻⁷ from level 12 to 13. At small
  nome (z ≤ 0.5, the four-point functions) both methods are accurate to 10⁻¹¹ or better. Consequences:
  * **Table S-vpp (disc VΨΨ).** The two channels now agree to 6·10⁻¹²–2·10⁻¹⁰ (2·10⁻⁹ at 11/35, limited by the
    modular kernel at b′² = 10/7), instead of the 10⁻⁷ quoted. At ĉ = −2 the values change by 1.5·10⁻¹⁰ from level 12
    to 13, by 3·10⁻¹¹ from 800 to 1200 points and by 5·10⁻¹¹ from ε = 0.5 to 0.3. The printed values are accurate
    only to between 3·10⁻⁸ and 3·10⁻⁶ (relative), and check_06 compares with them at 5·10⁻⁶. Improved values
    (G_i, which equals G_ii to the accuracy just stated):

    | ĉ | printed G_i / G_ii | this package |
    |---|---|---|
    | c = 28 | 2.616050347e−2 / 2.616050391e−2 | 2.616042704e−2 |
    | −2 | −3.815389907 / −3.815389941 | −3.815390779 |
    | 1 | −12.27660319 / −12.27660660 | −12.27660866 |
    | 0 | −8.363401364 / −8.363400122 | −8.363399851 |
    | −22/5 | −1.426896173 / −1.426896322 | −1.426895529 |
    | −7 | −0.4543253832 / −0.4543252249 | −0.4543251431 |
    | 11/35 | −9.440668494 / −9.440668401 | −9.44066684 |

  * **Table S-hzz.** Fit residuals 10⁻¹³ (paper 2–6·10⁻¹⁰); c_(1,3) factorizes to 2·10⁻⁸ (paper 10⁻⁴); c_(1,5)
    becomes resolvable and factorizes to 10⁻²; N1, c_(1,1), c_(1,3) agree with the printed values.
  * **Kac cancellation.** At bhat² = 1/√2 the ratios |T1+T2|/|T1| are 10⁻¹²–10⁻¹⁰ (paper up to 2·10⁻⁸).
* **Closed-form kernel vs Teschner–Vartanov integral** (Sec. S1 says 10⁻¹¹–10⁻¹³ at b² = 1, 1/2, 2/3, 2/5, 1/3):
  at the momenta of check_00 the agreement is 10⁻¹²–4·10⁻¹⁰, the closed form losing digits as mn grows (b² = 2/5);
  the integral is stable to 10⁻¹², as an evaluation with the independent `GeneralB` Γ_b confirms.
* **ĉ = 11/35 (bhat² = 5/7).** The modular kernel there comes from the closed-form fusion kernel at b′² = 10/7
  (mn = 70). At imaginary momenta it agrees with Teschner's integral to 2·10⁻¹², but with one momentum on the
  contour ε + iℝ it agrees with the independent Barnes-integral kernel (`ModularKernelGeneral`) only to
  3·10⁻⁹–3·10⁻⁸. This limits the 11/35 rows. The VΨΨ channels agree to 2·10⁻⁹. In the disc VVΨ the boundary channel
  carries a spurious real part ~2·10⁻⁸ and differs from the printed values by 8·10⁻⁹ (relative), so check_07 compares
  those three printed numbers at 3·10⁻⁸.
* **Irrational bhat² = 1/√2.** The Barnes-integral kernel agrees with the closed form to 10⁻¹²–10⁻¹⁰ (check_00,
  section 5). Boundary crossing at ĉ = 0.272 holds to 3·10⁻¹³, but G_s differs from the printed 5.947008604731 by
  7·10⁻¹¹ (relative), an error common to both channels. check_11 therefore compares the printed values of
  Tables S-4pt and S-cyl at 3·10⁻¹⁰.
* **Torus two-point Moore–Seiberg relation (S28)** at c = 28: with tilted lines the integrand decays only like
  e^{−πt² sin2θ/2}; the grid |t| ≤ 4 of the original script leaves 10⁻⁴; check_12 uses |t| ≤ 7 and gets ~10⁻⁸
  (paper: 1–4·10⁻⁷).
* **VVΨ (Table S-vvp)** uses the original comb-block code path (explicit inverses of the Gram matrices, kept so
  that the computation is the paper's). It reproduces all printed values to all digits except the three 11/35
  boundary-channel values above, and its accuracy is limited by the level N, as stated in the paper.

None of these affects a conclusion of the paper; all claimed agreements hold, most with better accuracy.

**How the tolerances were chosen.** Each test has an explicit tolerance printed next to the result. It is set by the
accuracy of the method (quadrature step, block level, contour cut-off, the 1e-12 of the closed-form kernels), with a
margin of at least 3 over what was observed in the reference runs; where a tolerance is looser than the observed
agreement for a physical reason (e.g. level-8 blocks in Table S-vvp), the script says so. Tests at the round-off floor
(a few in check_00 and check_01) may differ in the last digit on another platform (different BLAS or libm). Every line
prints both sides, so a marginal failure of that kind is easy to tell apart from a real one.

---

## 6. Provenance

The library was assembled from the project's scripts (claude/ folder of the project), with identical algorithms:
`barnes.py`, `gammab.py`, `gammab_gen.py` → `special.py`; `rtkernel.py`, `rtkernel_mn.py`, `bbb/fvec.py`, `modkernel.py`,
`gammab_gen.py` → `kernels.py`; `vblock.py`, `torusblock.py`, `bbb/ell.py` → `blocks.py`; `bbb/vir5.py`, `bbb/geom5.py`,
`bbb/amp5.py` → `comb5.py`; `bbb/data.py`, `bbb/data_fin.py` → `theory.py`. Changes: vectorisation over momenta,
the Gram-matrix solve (§5), a faster and slightly more accurate log G (shift to Re z ≥ 12 with a log-sum instead of
a log-Gamma sum), batched residue circles in the general-b kernel, coarser but equally accurate Gauss–Legendre panels
for Γ_b at arbitrary b.

| check | replaces |
|---|---|
| 01 | `testZ_standard.py exch/pole`, `bbb/testK2.py (b),(c)` |
| 02 | `testZ_standard.py pent/disc`, `bbb/testK2.py (a)`, `bbb/bb1_r2.py`, `bbb/t2rel_r2.py` |
| 03 | `testZ_standard.py four` (run there in the literal normalization and converted by 4√2) |
| 04 | `testQ_cylinder.py`, `testY_newnorm.py cyl` |
| 05 | `chk/disc2_fin.py` |
| 06 | `bbb/b1b2_amp.py`, `bbb/run_fin.py b1b2` |
| 07 | `bbb/amp5_test.py`, `chk/bb1_fin_fast.py`, `bbb/run_fin.py bb1` |
| 08 | `testU_annulus.py`, `testW_route2_annulus.py`, `check2_GS_annulus.py` |
| 09 | `psi0kernel.py`, `testZ_standard.py psi0`, `testN/O_psi0_*.py` |
| 10 | `testV_kac_cancel.py`, `testV2_cylinder_kac.py`, `testV3_disc_kac.py` |
| 11 | `testX_irrational.py` |
| 12 | `modkernel.py` (balancing), `bbb/t2rel.py`, `bbb/bb1_25.py` |
| 13 | `chk/hzz_disc.py`, `chk/hzz_disc2.py`, `chk/hzz_m2.py`, `bbb/halfzz*.py` |
| 14 | `bbb/testK2.py (d),(e)` |
| 15 | `check1_parity.py`, `testL_route1_generic.py` |

Not included: the later explorations in the project that are not part of the Letter (the on-shell lattice data,
minimal-model values, spectral-curve/complex-Liouville-string matching, the physical kernels in the mixed sector, the
discrete-crossing-identity proof; folders `chk/onshell`, `chk/mm`, `chk/cls`, `chk/r1cyl`, `chk/dci`, `chk/cosm*`).
