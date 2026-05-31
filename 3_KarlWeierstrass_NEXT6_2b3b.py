"""
3_KarlWeierstrass_NEXT6_2b3b — POGODNI deo iz 1_KarlWeierstrass_v2.py
Aparat 2b: Hurst / R-S analiza  +  Test 3b: Autokorelacija (ACF)

Self-contained:
  - KORAK 1: ucitavanje 4624 izvlacenja i izgradnja f(t) = lex-indeks
  - KORAK 2b: rolling/local Hurst (priprema)
  - KORAK 2b3b: ACF nad rolling H nizom, ACF nad f(t) kao kontrola,
                Ljung-Box i shuffled max|ACF| referenca

Output:
  3_KarlWeierstrass_NEXT6_2b3b.png
  3_KarlWeierstrass_NEXT6_2b3b.txt
"""

import csv
import math
import os
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


T0 = time.time()

CSV_DRAWS = "/Users/4c/Desktop/GHQ/data/loto7_4624_k43.csv"

HERE = os.path.dirname(os.path.abspath(__file__))
PNG_PATH = os.path.join(HERE, "3_KarlWeierstrass_NEXT6_2b3b.png")
TXT_PATH = os.path.join(HERE, "3_KarlWeierstrass_NEXT6_2b3b.txt")

N_MAX = 39
K_PICK = 7
TOTAL_COMBOS = math.comb(N_MAX, K_PICK)


# ─── helperi (samo oni potrebni za 2b + 2b3b) ────────────────────────
def read_loto_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < K_PICK:
                continue
            try:
                nums = tuple(sorted(int(x) for x in row[:K_PICK]))
            except ValueError:
                continue
            if len(nums) == K_PICK and len(set(nums)) == K_PICK:
                rows.append(nums)
    return rows


def lex_rank_1based(combo, n=N_MAX, k=K_PICK):
    """1-based lex indeks (poklapa se sa rednim brojem u kombinacije_39C7.csv)."""
    combo = tuple(sorted(combo))
    rank0 = 0
    prev = 0
    for i, value in enumerate(combo):
        remaining = k - i - 1
        for candidate in range(prev + 1, value):
            rank0 += math.comb(n - candidate, remaining)
        prev = value
    return rank0 + 1


def hurst_rs(series, min_window=8, max_window=None):
    """R/S Hurst procena: slope log(R/S) prema log(window)."""
    x = np.asarray(series, dtype=float)
    n = len(x)
    if max_window is None:
        max_window = max(min_window * 2, n // 4)

    windows = []
    w = min_window
    while w <= max_window:
        windows.append(w)
        w = int(w * 1.45) + 1

    used_windows = []
    rs_values = []
    for w in windows:
        chunks = n // w
        if chunks < 2:
            continue
        vals = []
        for i in range(chunks):
            seg = x[i * w:(i + 1) * w]
            y = seg - seg.mean()
            z = np.cumsum(y)
            r = z.max() - z.min()
            s = seg.std(ddof=1)
            if s > 0:
                vals.append(r / s)
        if vals:
            used_windows.append(w)
            rs_values.append(float(np.mean(vals)))

    used_windows = np.asarray(used_windows, dtype=float)
    rs_values = np.asarray(rs_values, dtype=float)
    slope, intercept = np.polyfit(np.log(used_windows), np.log(rs_values), 1)
    fit = intercept + slope * np.log(used_windows)
    ss_res = float(np.sum((np.log(rs_values) - fit) ** 2))
    ss_tot = float(np.sum((np.log(rs_values) - np.log(rs_values).mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), float(r2), used_windows, rs_values


def rolling_hurst_rs(series, window=768, step=128):
    """Rolling R/S Hurst procena kroz vreme."""
    x = np.asarray(series, dtype=float)
    centers = []
    hvals = []
    r2vals = []
    for start in range(0, len(x) - window + 1, step):
        seg = x[start:start + window]
        h, _, r2, _, _ = hurst_rs(seg, min_window=8, max_window=max(32, window // 4))
        centers.append(start + window // 2 + 1)
        hvals.append(h)
        r2vals.append(r2)
    return (
        np.asarray(centers, dtype=float),
        np.asarray(hvals, dtype=float),
        np.asarray(r2vals, dtype=float),
    )


def autocorr_values(series, max_lag=60):
    """ACF za lagove 0..max_lag."""
    x = np.asarray(series, dtype=float)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    vals = []
    for lag in range(max_lag + 1):
        if lag == 0:
            vals.append(1.0)
        else:
            vals.append(float(np.dot(x[:-lag], x[lag:]) / denom))
    return np.asarray(vals, dtype=float)


def ljung_box_approx(acf_vals, n, h):
    """Ljung-Box Q aproksimacija za prvih h ACF lagova."""
    lags = np.arange(1, h + 1, dtype=float)
    q = n * (n + 2) * float(np.sum((acf_vals[1:h + 1] ** 2) / (n - lags)))
    p = float(stats.chi2.sf(q, h))
    return float(q), p


# ─── KORAK 1: f(t) = lex-indeks ──────────────────────────────────────
draws = read_loto_csv(CSV_DRAWS)
N = len(draws)
lex_idx = np.array([lex_rank_1based(c) for c in draws], dtype=np.float64)

print()
print("3_KarlWeierstrass_NEXT6_2b3b — KORAK 1: formiranje krive f(t)")
print(f"  CSV:                  {CSV_DRAWS}")
print(f"  Ucitano izvlacenja:    {N}")
print(f"  C(39,7):              {TOTAL_COMBOS:,}")
print()

with open(TXT_PATH, "w", encoding="utf-8") as f:
    f.write("3_KarlWeierstrass_NEXT6_2b3b — Hurst/R-S + ACF (POGODNO)\n")
    f.write("=" * 60 + "\n\n")
    f.write("KORAK 1: Weierstrass-ova funkcija nad svih izvucenih kombinacija\n\n")
    f.write(f"  CSV izvucenih:        {CSV_DRAWS}\n")
    f.write(f"  Ucitano izvlacenja:    {N}\n")
    f.write(f"  C(39,7):              {TOTAL_COMBOS:,}\n")
    f.write("  f(t) = lex-indeks izvucene kombinacije u skupu svih 39C7\n\n")


# ─── KORAK 2b: rolling/local Hurst (priprema) ────────────────────────
rolling_window = 768
rolling_step = 128
roll_centers, roll_h, roll_r2 = rolling_hurst_rs(
    lex_idx, window=rolling_window, step=rolling_step
)


# ─── KORAK 2b3b: ACF nad rolling H + kontrola + shuffled referenca ──
T0_2B3B = time.time()

acf_max_lag = 60
roll_acf_max_lag = min(20, max(1, len(roll_h) - 2))
roll_acf_lags = np.arange(0, roll_acf_max_lag + 1)
acf_roll_h = autocorr_values(roll_h, roll_acf_max_lag)
acf_f_control = autocorr_values(lex_idx, acf_max_lag)

roll_acf_band = 1.96 / np.sqrt(len(roll_h))
roll_acf_body = acf_roll_h[1:]
roll_max_abs_acf = float(np.max(np.abs(roll_acf_body)))
roll_sig_lag_count = int(np.sum(np.abs(roll_acf_body) > roll_acf_band))
roll_top_idx = np.argsort(np.abs(roll_acf_body))[-min(10, len(roll_acf_body)):][::-1] + 1
roll_top_acf_pairs = [(int(lag), float(acf_roll_h[lag])) for lag in roll_top_idx]

roll_lb_h = min(10, roll_acf_max_lag)
roll_lb_q, roll_lb_p = ljung_box_approx(acf_roll_h, len(roll_h), roll_lb_h)

rng_2b3b = np.random.default_rng(47)
roll_acf_shuffle_runs = 500
shuffle_roll_max_abs_acf = []
for _ in range(roll_acf_shuffle_runs):
    shuffled_h = rng_2b3b.permutation(roll_h)
    shuffled_acf = autocorr_values(shuffled_h, roll_acf_max_lag)
    shuffle_roll_max_abs_acf.append(float(np.max(np.abs(shuffled_acf[1:]))))
shuffle_roll_max_abs_acf = np.asarray(shuffle_roll_max_abs_acf, dtype=float)
shuffle_roll_acf_mean = float(shuffle_roll_max_abs_acf.mean())
shuffle_roll_acf_std = float(shuffle_roll_max_abs_acf.std(ddof=1))
shuffle_roll_acf_p = float(np.mean(shuffle_roll_max_abs_acf >= roll_max_abs_acf))
shuffle_roll_acf_z = (
    (roll_max_abs_acf - shuffle_roll_acf_mean) / (shuffle_roll_acf_std + 1e-12)
)

if roll_lb_p <= 0.05 or shuffle_roll_acf_p <= 0.05:
    roll_acf_note = "rolling H ima ACF signal iznad shuffled reference"
else:
    roll_acf_note = "rolling H nema jak ACF signal iznad shuffled reference"

print()
print("KORAK 2b3b: Aparat 2b Hurst/R-S + Test 3b Autokorelacija (ACF)")
print(f"  rolling H max |ACF| lag 1..{roll_acf_max_lag}: {roll_max_abs_acf:.4f}")
print(f"  95% band: +/-{roll_acf_band:.4f}   znacajnih lagova: "
      f"{roll_sig_lag_count}/{roll_acf_max_lag}")
print(f"  Ljung-Box aproks. h={roll_lb_h}: Q={roll_lb_q:.2f}  p={roll_lb_p:.4f}")
print(f"  shuffled max|ACF|: mean={shuffle_roll_acf_mean:.4f} std={shuffle_roll_acf_std:.4f} "
      f"z={shuffle_roll_acf_z:.2f} p={shuffle_roll_acf_p:.4f}")
print(f"  ⇒ {roll_acf_note}")
print()

fig2b3b, ax2b3b = plt.subplots(1, 3, figsize=(16, 5))
fig2b3b.suptitle("KORAK 2b3b: Hurst/R-S aparat + ACF test  (POGODNO)",
                 fontsize=13, fontweight="bold")

ax2b3b[0].bar(roll_acf_lags[1:], acf_roll_h[1:], width=0.8, color="darkslateblue")
ax2b3b[0].axhline(roll_acf_band, color="crimson", linestyle="--", linewidth=1.2)
ax2b3b[0].axhline(-roll_acf_band, color="crimson", linestyle="--", linewidth=1.2)
ax2b3b[0].axhline(0, color="black", linewidth=0.6)
ax2b3b[0].set_title("ACF rolling/local Hurst niza")
ax2b3b[0].set_xlabel("lag")
ax2b3b[0].set_ylabel("ACF")

ax2b3b[1].bar(np.arange(1, acf_max_lag + 1), acf_f_control[1:], width=0.8,
              color="steelblue")
ax2b3b[1].axhline(1.96 / np.sqrt(N), color="crimson", linestyle="--", linewidth=1.2)
ax2b3b[1].axhline(-1.96 / np.sqrt(N), color="crimson", linestyle="--", linewidth=1.2)
ax2b3b[1].axhline(0, color="black", linewidth=0.6)
ax2b3b[1].set_title("Kontrola: ACF f(t)")
ax2b3b[1].set_xlabel("lag")
ax2b3b[1].set_ylabel("ACF")

ax2b3b[2].hist(shuffle_roll_max_abs_acf, bins=24, color="lightgray", edgecolor="white")
ax2b3b[2].axvline(roll_max_abs_acf, color="crimson", linewidth=2,
                  label=f"observed={roll_max_abs_acf:.3f}")
ax2b3b[2].axvline(shuffle_roll_acf_mean, color="black", linestyle="--",
                  label=f"shuffle mean={shuffle_roll_acf_mean:.3f}")
ax2b3b[2].set_title("Shuffled rolling H max |ACF|")
ax2b3b[2].set_xlabel("max |ACF|")
ax2b3b[2].set_ylabel("broj")
ax2b3b[2].legend(fontsize=8)

for a in ax2b3b:
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.grid(True, alpha=0.2)

fig2b3b.tight_layout()
fig2b3b.savefig(PNG_PATH, dpi=150, bbox_inches="tight")
plt.show()

with open(TXT_PATH, "a", encoding="utf-8") as f:
    f.write("\n")
    f.write("=" * 60 + "\n")
    f.write("KORAK 2b3b: Aparat 2b Hurst/R-S + Test 3b Autokorelacija (ACF)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"  PNG:                  {PNG_PATH}\n\n")
    f.write("ACF nad rolling/local Hurst nizom:\n")
    f.write(f"  broj rolling H tacaka = {len(roll_h)}\n")
    f.write(f"  max lag               = {roll_acf_max_lag}\n")
    f.write(f"  95% band              = +/-{roll_acf_band:.6f}\n")
    f.write(f"  max |ACF|             = {roll_max_abs_acf:.6f}\n")
    f.write(f"  znacajnih lagova      = {roll_sig_lag_count}/{roll_acf_max_lag}\n")
    f.write(f"  Ljung-Box h           = {roll_lb_h}\n")
    f.write(f"  Ljung-Box Q           = {roll_lb_q:.6f}\n")
    f.write(f"  Ljung-Box p           = {roll_lb_p:.6f}\n\n")
    f.write("Shuffled rolling H max |ACF| referenca:\n")
    f.write(f"  runs                  = {roll_acf_shuffle_runs}\n")
    f.write(f"  mean                  = {shuffle_roll_acf_mean:.6f}\n")
    f.write(f"  std                   = {shuffle_roll_acf_std:.6f}\n")
    f.write(f"  z                     = {shuffle_roll_acf_z:.6f}\n")
    f.write(f"  p(shuffled >= obs)    = {shuffle_roll_acf_p:.6f}\n")
    f.write(f"  interpret.            = {roll_acf_note}\n\n")
    f.write("Top ACF lagovi rolling H po apsolutnoj vrednosti:\n")
    f.write(f"  {'lag':<8}{'ACF':>16}\n")
    for lag, val in roll_top_acf_pairs:
        f.write(f"  {lag:<8}{val:>16,.8f}\n")
    f.write("\n")

    elapsed_2b3b = time.time() - T0_2B3B
    f.write(f"Vreme KORAKA 2b3b: {timedelta(seconds=int(elapsed_2b3b))} ({elapsed_2b3b:.1f} s)\n")
    f.write(f"Ukupno vreme:       {timedelta(seconds=int(time.time()-T0))} ({time.time()-T0:.1f} s)\n")

print(f"PNG saved → {PNG_PATH}")
print(f"TXT saved → {TXT_PATH}")
print(f"Vreme KORAKA 2b3b: {timedelta(seconds=int(time.time()-T0_2B3B))} "
      f"({time.time()-T0_2B3B:.1f} s)")
print(f"Ukupno vreme:      {timedelta(seconds=int(time.time()-T0))} "
      f"({time.time()-T0:.1f} s)")
print()
print("KRAJ 3_KarlWeierstrass_NEXT6_2b3b.")
print()
"""
3_KarlWeierstrass_NEXT6_2b3b — KORAK 1: formiranje krive f(t)
  CSV:                  /Users/4c/Desktop/GHQ/data/loto7_4624_k43.csv
  Ucitano izvlacenja:    4624
  C(39,7):              15,380,937


KORAK 2b3b: Aparat 2b Hurst/R-S + Test 3b Autokorelacija (ACF)
  rolling H max |ACF| lag 1..20: 0.4771
  95% band: +/-0.3520   znacajnih lagova: 1/20
  Ljung-Box aproks. h=10: Q=21.93  p=0.0154
  shuffled max|ACF|: mean=0.3101 std=0.0738 z=2.26 p=0.0280
  ⇒ rolling H ima ACF signal iznad shuffled reference

PNG saved → /Users/4c/Desktop/GHQ/KarlWeierstrass/3_KarlWeierstrass_NEXT6_2b3b.png
TXT saved → /Users/4c/Desktop/GHQ/KarlWeierstrass/3_KarlWeierstrass_NEXT6_2b3b.txt
Vreme KORAKA 2b3b: 0:00:21 (21.9 s)
Ukupno vreme:      0:00:22 (22.1 s)

KRAJ 3_KarlWeierstrass_NEXT6_2b3b.
"""



###############   PREDIKCIJA 6  ###############################

"""
NEXT6 (2b3b, rolling H ACF) — AR(1) na rolling H, pa drift u tom režimu.
"""


def lex_unrank_1based(rank, n=N_MAX, k=K_PICK):
    """Vracanje 1-based lex indeksa u Loto 7/39 kombinaciju."""
    rank0 = int(rank) - 1
    combo = []
    prev = 0
    for i in range(k):
        remaining = k - i - 1
        for candidate in range(prev + 1, n + 1):
            count = math.comb(n - candidate, remaining)
            if rank0 >= count:
                rank0 -= count
            else:
                combo.append(candidate)
                prev = candidate
                break
    return tuple(combo)


T0_PRED6 = time.time()

# Rolling H ima ACF signal. Prvo predvidjamo sledeci H AR(1) logikom,
# pa tim H rezimom skaliramo lokalni drift u zadnjem rolling prozoru.
rho_h = float(acf_roll_h[1]) if len(acf_roll_h) > 1 else 0.0
roll_h_mean_value = float(roll_h.mean())
last_h = float(roll_h[-1])
pred_h = roll_h_mean_value + rho_h * (last_h - roll_h_mean_value)
pred_h = float(np.clip(pred_h, 0.0, 1.0))

local_window = rolling_window
local_y = np.asarray(lex_idx[-local_window:], dtype=float)
local_x = np.arange(len(local_y), dtype=float)
local_slope, local_intercept = np.polyfit(local_x, local_y, 1)
local_fit = local_intercept + local_slope * local_x
local_resid = local_y - local_fit
local_resid_std = float(local_resid.std(ddof=1))

recent_incr = np.diff(local_y)
recent_mean_incr = float(recent_incr.mean())
last_lex = float(lex_idx[-1])
last_incr = float(np.diff(lex_idx)[-1])

h_strength = float(np.clip((pred_h - 0.5) / 0.15, 0.0, 1.0))
pred_incr = (1.0 - h_strength) * recent_mean_incr + h_strength * last_incr
pred_lex_float = last_lex + pred_incr
pred_lex = int(np.clip(round(pred_lex_float), 1, TOTAL_COMBOS))
pred_combo = lex_unrank_1based(pred_lex)

z_grid = [-1.28, -0.84, -0.43, 0.0, 0.43, 0.84, 1.28]
candidate_rows = []
seen_lex = set()
for z in z_grid:
    cand_lex = int(np.clip(round(pred_lex_float + z * local_resid_std), 1, TOTAL_COMBOS))
    if cand_lex in seen_lex:
        continue
    seen_lex.add(cand_lex)
    candidate_rows.append((z, cand_lex, lex_unrank_1based(cand_lex)))

print()
print("PREDIKCIJA 6 — NEXT6 / 2b3b / rolling H ACF")
print(f"  rho_H lag-1            = {rho_h:.8f}")
print(f"  rolling H mean         = {roll_h_mean_value:.8f}")
print(f"  zadnji rolling H       = {last_h:.8f}")
print(f"  pred. rolling H        = {pred_h:.8f}")
print(f"  H strength             = {h_strength:.6f}")
print(f"  lokalni slope          = {local_slope:,.2f}")
print(f"  recent mean dX         = {recent_mean_incr:,.2f}")
print(f"  zadnji dX              = {last_incr:,.2f}")
print(f"  pred. inkrement        = {pred_incr:,.2f}")
print(f"  pred. lex              = {pred_lex:,}")
print(f"  pred. kombinacija      = {pred_combo}")
print("  kandidati oko rolling-H ACF prognoze:")
for z, cand_lex, combo in candidate_rows:
    print(f"    z={z:>5.2f}  lex={cand_lex:>10,}  combo={combo}")
print()

with open(TXT_PATH, "a", encoding="utf-8") as f:
    f.write("\n")
    f.write("=" * 60 + "\n")
    f.write("PREDIKCIJA 6: NEXT6 / 2b3b / rolling H ACF\n")
    f.write("=" * 60 + "\n\n")
    f.write("Model:\n")
    f.write("  Rolling H ima ACF signal iznad shuffled reference.\n")
    f.write("  Sledeci H se procenjuje AR(1): H_next = mean(H) + rho_H*(H_last-mean(H)).\n")
    f.write("  Prediktovani H zatim odredjuje koliko zadnji inkrement nosi trend.\n\n")
    f.write("Parametri:\n")
    f.write(f"  rho_H lag-1            = {rho_h:.8f}\n")
    f.write(f"  rolling H mean         = {roll_h_mean_value:.8f}\n")
    f.write(f"  zadnji rolling H       = {last_h:.8f}\n")
    f.write(f"  pred. rolling H        = {pred_h:.8f}\n")
    f.write(f"  H strength             = {h_strength:.8f}\n")
    f.write(f"  local window           = {local_window}\n")
    f.write(f"  lokalni slope          = {local_slope:,.8f}\n")
    f.write(f"  lokalni resid std      = {local_resid_std:,.8f}\n")
    f.write(f"  recent mean dX         = {recent_mean_incr:,.8f}\n")
    f.write(f"  zadnji dX              = {last_incr:,.8f}\n")
    f.write(f"  zadnji lex             = {int(last_lex):,}\n")
    f.write(f"  pred. inkrement        = {pred_incr:,.8f}\n\n")
    f.write("Glavna prognoza:\n")
    f.write(f"  pred. lex float        = {pred_lex_float:,.8f}\n")
    f.write(f"  pred. lex              = {pred_lex:,}\n")
    f.write(f"  pred. kombinacija      = {pred_combo}\n\n")
    f.write("Kandidati oko rolling-H ACF prognoze:\n")
    f.write(f"  {'z':>8}{'lex':>14}  kombinacija\n")
    for z, cand_lex, combo in candidate_rows:
        f.write(f"  {z:>8.2f}{cand_lex:>14,}  {combo}\n")
    f.write("\n")
    elapsed_pred6 = time.time() - T0_PRED6
    f.write(f"Vreme PREDIKCIJE 6: {timedelta(seconds=int(elapsed_pred6))} ({elapsed_pred6:.1f} s)\n")

print(f"TXT updated → {TXT_PATH}")
print(f"Vreme PREDIKCIJE 6: {timedelta(seconds=int(time.time()-T0_PRED6))} "
      f"({time.time()-T0_PRED6:.1f} s)")
print()


"""
Predikcija će koristiti ACF/AR(1) nad rolling H režimom da proceni sledeći H, pa time skalira lokalni drift.

Dodajem PREDIKCIJU 6: AR(1) procena sledećeg rolling H iz ACF-a, zatim lokalni drift u zadnjem prozoru prema procenjenom H režimu.



Dodao sam PREDIKCIJA 6:

AR(1) procena sledećeg rolling H preko rho_H = ACF lag-1
skaliranje lokalnog drift-a prema procenjenom H_next
glavna Loto 7/39 kombinacija + kandidati oko prognoze
upis u 3_KarlWeierstrass_NEXT6_2b3b.txt
"""



"""
3_KarlWeierstrass_NEXT6_2b3b — KORAK 1: formiranje krive f(t)
  CSV:                  /Users/4c/Desktop/GHQ/data/loto7_4624_k43.csv
  Ucitano izvlacenja:    4624
  C(39,7):              15,380,937


KORAK 2b3b: Aparat 2b Hurst/R-S + Test 3b Autokorelacija (ACF)
  rolling H max |ACF| lag 1..20: 0.4771
  95% band: +/-0.3520   znacajnih lagova: 1/20
  Ljung-Box aproks. h=10: Q=21.93  p=0.0154
  shuffled max|ACF|: mean=0.3101 std=0.0738 z=2.26 p=0.0280
  ⇒ rolling H ima ACF signal iznad shuffled reference

PNG saved → /Users/4c/Desktop/GHQ/KarlWeierstrass/3_KarlWeierstrass_NEXT6_2b3b.png
TXT saved → /Users/4c/Desktop/GHQ/KarlWeierstrass/3_KarlWeierstrass_NEXT6_2b3b.txt
Vreme KORAKA 2b3b: 0:00:03 (3.5 s)
Ukupno vreme:      0:00:03 (3.6 s)

KRAJ 3_KarlWeierstrass_NEXT6_2b3b.


PREDIKCIJA 6 — NEXT6 / 2b3b / rolling H ACF
  rho_H lag-1            = 0.47712746
  rolling H mean         = 0.59744976
  zadnji rolling H       = 0.62382045
  pred. rolling H        = 0.61003194
  H strength             = 0.733546
  lokalni slope          = -739.26
  recent mean dX         = -6,859.23
  zadnji dX              = -2,143,496.00
  pred. inkrement        = -1,574,181.20
  pred. lex              = 1
  pred. kombinacija      = (1, 2, 3, 4, 5, 6, 7)
  kandidati oko rolling-H ACF prognoze:
    z=-1.28  lex=         1  combo=(1, 2, 3, 4, 5, 6, 7)
    z= 0.43  lex=   874,227  combo=(1, 4, 6, 10, 15, 20, 27)
    z= 0.84  lex= 2,719,507  combo=(1, 19, 25, 28, 31, 32, 34)
    z= 1.28  lex= 4,699,808  combo=(2, 11, 21, 28, 34, 35, 36)

TXT updated → /Users/4c/Desktop/GHQ/KarlWeierstrass/3_KarlWeierstrass_NEXT6_2b3b.txt
Vreme PREDIKCIJE 6: 0:00:00 (0.0 s)
"""
