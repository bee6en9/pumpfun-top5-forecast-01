#!/usr/bin/env python3
"""pumpfun-top5-forecast-01 entry: momentum-aware forecaster (v3).

Reads /input/history.ndjson (10-minute snapshots of pump.fun coins launched
since the opening, ranked by market cap) plus /input/task.json, and writes one
forecast to /output/forecast.json:
  {probabilities: [{mint, place 1-5, p}], top5: [5 mints in order],
   sizes: [{place 1-5, bin 0-81, p}]}
Python 3.13 stdlib only, no network, deterministic (one fixed-seed RNG).

Method: per coin, fit the recent log10 market-cap drift (slope over the last
up-to-4 marks blended with the last step) and extrapolate it, damped, toward
the answer reads — momentum matters over minutes, not over 10 hours. Rank
candidates by that predicted cap; each place's probabilities are a Gaussian
over predicted-rank distance, with a large share left unnamed because
newcomers take a large share of top-5 places (3 of the last rehearsal read's
top 5 were never in the history).

v3 changes the SIZE model (v2's weakness: practice size rank 328/615 vs
order rank 199/615). v2 put one symmetric normal per place around the
predicted k-th coin's cap. That is wrong twice: the top-5 caps are set by
NEWCOMERS that pump far above any coin in the history (at every rehearsal
mark the #1 cap equals the largest cap among coins born in the last 3
marks), and a normal's left tail dumps up to 38% of a place's mass into
bin 0 (<100 SOL), where a top-5 cap never lands. v3 replaces it with a
Monte Carlo: simulate the field at each of the 7 answer reads — every
candidate coin's log-cap as its drift path plus a random-walk term from its
own measured per-mark volatility (dispersion saturating with horizon,
mean-reverting), plus a newcomer process (births per mark from the history,
a Poisson count of mooners, each one's cap drawn around the
trend-extrapolated top level at the read, minus an exponential gap) — take
the top 5 of each simulated field, bin them, and pool the histograms over
the reads. The result is an honest, right-skewed size distribution per
place instead of one symmetric normal. All constants were calibrated on a
local backtest of the rehearsal history (see dev/): the forecaster beats
the house persistence baseline on every split.
"""
import bisect
import json
import math
import os
import random

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"

# --- tunables (calibrated on the rehearsal backtest; see dev/) ---
DRIFT_CLIP = 0.15        # max per-mark log10 drift credited to a coin
DRIFT_FLOOR = -0.06      # max per-mark log10 decay credited
H_EFF_CAP = 6.0          # extrapolate drift for at most this many marks
H_DECAY = 12.0           # horizon (marks) over which drift influence halves-ish
SIZE_SD_BASE = 0.35      # sd of a place's log10 cap, at zero horizon
SIZE_SD_PER_MARK = 0.012 # extra sd per mark of horizon
SIZE_SD_MAX = 0.85
CENTER_SHIFT1 = 0.20     # log10 lift of place 1's size centre (order statistic:
                         # the actual cap at a place is the k-th largest of many
                         # coins, so it sits above the k-th predicted coin's mu);
                         # tapers linearly to 0 at place 5
RANK_SPREAD_BASE = 2.1   # sd (in ranks) of place probability around predicted rank
RANK_SPREAD_PER_MARK = 0.02
RANK_SPREAD_MAX = 3.5
UNNAMED_BASE = 0.40      # share of each place left unnamed (newcomers)
UNNAMED_PER_MARK = 0.004 # extra unnamed share per mark of horizon
UNNAMED_MAX = 0.55
N_CANDIDATES = 30        # coins ranked by predicted mu considered per place
VENUE_ADJ = {"pool": 0.0, "migrating": 0.05, "curve": -0.05}

# --- Monte Carlo size model (v3) ---
MC_SEED = 20261112        # fixed seed: two runs must be byte-identical
MC_N_SIM = 4000           # simulated fields per answer read
MC_PROFILE_AGES = 6       # young-coin profiles tracked to this age (marks)
MC_MOONER_MIN = 2.8       # a profile is a "mooner" if its max log-cap >= this
MC_MOONER_FLOOR = 5       # keep at least this many mooner profiles (top by max)
MC_VOL_SHRINK = 4.0       # pseudo-observations shrinking a coin's vol to pooled
MC_VOL_MIN = 0.05         # per-mark log-cap volatility floor
MC_VOL_MAX = 0.40         # per-mark log-cap volatility cap
MC_VOL_HORIZON_CAP = 9.0   # random-walk dispersion saturates after this many marks
MC_VOL_SD_CAP = 0.35      # cap on a coin's total log-cap sd at an answer read
MC_BIN_ALPHA = 0.5        # Dirichlet pseudo-count per size bin (smoothing)
MC_BIRTH_FLOOR = 1.0      # floor on births per mark for the newcomer process
# --- level anchor for the newcomer caps (v3.1) ---
# The profile-resampled mooner caps are biased low when the history is
# short and the market is rising (split A: the future top-1 cap 3.56 sat
# 0.45 above every cap in the 3-mark history). So a mooner's cap is drawn
# around the TREND-EXTRAPOLATED top level at the read, not around a past
# profile's cap: the level is the forward-looking quantity.
MC_GAP_MEAN = 0.18        # mean gap of a mooner's cap below the top level
MC_LEVEL_G_MAX = 0.6      # cap on the top level's total growth (log10)
MC_LEVEL_G_CLIP = 0.08    # clip on the top level's per-mark growth
MC_LEVEL_G_FLOOR = -0.02  # the top level rarely falls fast in this market
MC_LEVEL_SIGMA_BASE = 0.08 # sd of the top level at zero horizon
MC_LEVEL_SIGMA_PER_MARK = 0.03
MC_LEVEL_SIGMA_CAP = 0.40


def logcap(row):
    """log10 of a row's market cap in SOL."""
    return math.log10(int(row["market_cap_lamports"])) - 9


def parse_time(s):
    """ISO 8601 UTC (with millis and Z) -> epoch seconds (float)."""
    # format: 2026-11-11T20:00:04.073Z
    y = int(s[0:4]); mo = int(s[5:7]); d = int(s[8:10])
    hh = int(s[11:13]); mm = int(s[14:16]); sec = float(s[17:19])
    if len(s) > 20 and s[19] == ".":
        sec += float("0." + s[20:-1]) if s.endswith("Z") else 0.0
    import datetime
    dt = datetime.datetime(y, mo, d, hh, mm, 0, tzinfo=datetime.timezone.utc)
    return dt.timestamp() + sec


def load_history(path):
    marks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list):
                marks.append(m)
    return marks


def coin_features(marks):
    """Per-mint time series from the marks.

    Returns {mint: dict(last_log, last_rank, last_venue, first_idx, last_idx,
    slope, g1, n_seen)} using only marks where the coin was ranked.
    """
    series = {}
    for i, m in enumerate(marks):
        for r in m.get("rows", []):
            d = series.get(r["mint"])
            if d is None:
                d = series[r["mint"]] = {"pts": [], "venue": r["venue"]}
            d["pts"].append((i, logcap(r), int(r["rank"])))
            d["venue"] = r["venue"]
    out = {}
    # pooled per-mark volatility of log-cap changes (top-50 coins only: the
    # field's small coins barely move and would drag the pool to zero).
    pool_deltas = []
    for mint, d in series.items():
        pts = d["pts"]
        if len(pts) >= 2 and pts[-1][2] <= 50:
            for a, b in zip(pts, pts[1:]):
                if b[0] == a[0] + 1:
                    pool_deltas.append(b[1] - a[1])
    if pool_deltas:
        pm = sum(pool_deltas) / len(pool_deltas)
        pool_var = sum((x - pm) ** 2 for x in pool_deltas) / len(pool_deltas)
    else:
        pool_var = 0.04  # fallback sd 0.20
    pool_sd = math.sqrt(max(pool_var, 1e-6))
    for mint, d in series.items():
        pts = d["pts"]
        last_i, last_log, last_rank = pts[-1]
        first_i = pts[0][0]
        # slope of log-cap over the last up-to-4 observations (per mark)
        tail = pts[-4:]
        slope = 0.0
        if len(tail) >= 2:
            xs = [p[0] for p in tail]
            ys = [p[1] for p in tail]
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            den = sum((x - mx) ** 2 for x in xs)
            if den > 0:
                slope = num / den
        g1 = 0.0
        if len(pts) >= 2:
            g1 = pts[-1][1] - pts[-2][1]
        # per-mark volatility: sd of this coin's consecutive log-cap changes,
        # shrunk toward the pooled sd (young coins have too few deltas).
        deltas = [b[1] - a[1] for a, b in zip(pts, pts[1:]) if b[0] == a[0] + 1]
        if deltas:
            dm = sum(deltas) / len(deltas)
            var = sum((x - dm) ** 2 for x in deltas) / len(deltas)
            n = len(deltas)
            var = (n * var + MC_VOL_SHRINK * pool_var) / (n + MC_VOL_SHRINK)
        else:
            var = pool_var
        sigma = min(MC_VOL_MAX, max(MC_VOL_MIN, math.sqrt(max(var, 1e-6))))
        out[mint] = {
            "last_log": last_log, "last_rank": last_rank,
            "venue": d["venue"], "first_idx": first_i, "last_idx": last_i,
            "slope": slope, "g1": g1, "n_seen": len(pts), "sigma": sigma,
        }
    return out


def newcomer_stats(marks):
    """Empirical newcomer (birth) process for the Monte Carlo size model.

    Returns {"births_per_mark", "mooner_profiles", "f_moon"} where a profile
    is one young coin's log-cap by age (marks since first seen, ages
    0..MC_PROFILE_AGES; None where it was not ranked), and a mooner profile
    is one whose max cap reaches MC_MOONER_MIN (or, if too few do, the top
    MC_MOONER_FLOOR profiles by max cap). The top-5 caps at every rehearsal
    mark are set by coins born in the last few marks, so the size model
    must draw the newcomers' caps from these observed profiles.
    """
    n = len(marks)
    first_seen = {}
    caps_by_age = {}
    for i, m in enumerate(marks):
        for r in m.get("rows", []):
            mint = r["mint"]
            if mint not in first_seen:
                first_seen[mint] = i
                caps_by_age[mint] = [None] * (MC_PROFILE_AGES + 1)
            age = i - first_seen[mint]
            if age <= MC_PROFILE_AGES:
                caps_by_age[mint][age] = logcap(r)
    births = sum(1 for mint, f in first_seen.items() if f >= 1)
    births_per_mark = max(MC_BIRTH_FLOOR,
                          births / max(1, n - 1)) if n > 1 else MC_BIRTH_FLOOR
    profiles = []
    for mint in sorted(first_seen):
        caps = caps_by_age[mint]
        present = [c for c in caps if c is not None]
        if not present:
            continue
        profiles.append((max(present), caps))
    profiles.sort(key=lambda t: -t[0])
    mooners = [caps for mx, caps in profiles if mx >= MC_MOONER_MIN]
    if len(mooners) < MC_MOONER_FLOOR:
        mooners = [caps for _, caps in profiles[:MC_MOONER_FLOOR]]
    f_moon = len(mooners) / max(1, len(profiles))
    return {"births_per_mark": births_per_mark,
            "mooner_profiles": mooners, "f_moon": f_moon,
            "n_profiles": len(profiles)}


def predict_mu(feat, last_idx, horizon_marks):
    """Predicted log10 cap at the middle answer read.

    Extrapolates the recent drift, but only for a few marks and with the
    influence decaying as the horizon grows: a coin that pumped in the last
    hour will not keep pumping for 10 hours.
    """
    drift = 0.6 * feat["slope"] + 0.4 * feat["g1"]
    drift = max(DRIFT_FLOOR, min(DRIFT_CLIP, drift))
    h_eff = min(horizon_marks, H_EFF_CAP)
    decay = 1.0 / (1.0 + horizon_marks / H_DECAY)
    mu = feat["last_log"] + drift * h_eff * decay
    mu += VENUE_ADJ.get(feat["venue"], 0.0)
    # coins that vanished from the last marks are fading, not rising
    gap = last_idx - feat["last_idx"]
    if gap > 0:
        mu -= 0.15 * gap
    return mu


def _poisson(rng, lam):
    """A Poisson(lam) draw from a seeded RNG (Knuth for small lam)."""
    if lam <= 0:
        return 0
    if lam <= 30.0:
        limit = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            p *= rng.random()
            if p <= limit:
                return k - 1
    return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))


def top_level_forecast(marks):
    """Trend of the market's top level (the #1 coin's log-cap).

    Returns (L_last, g): the top-1 log-cap at the last mark and its
    per-mark growth, an OLS slope over the last up-to-6 marks, clipped to
    [MC_LEVEL_G_FLOOR, MC_LEVEL_G_CLIP]. The top level is what the
    newcomers' caps anchor to: a newborn that moons lands just below the
    level the market has reached by the read, not at a cap some earlier
    coin happened to reach.
    """
    tops = []
    for m in marks:
        rows = m.get("rows", [])
        if rows:
            tops.append(max(logcap(r) for r in rows))
    if not tops:
        return 1.45, 0.0
    L_last = tops[-1]
    tail = tops[-6:]
    if len(tail) >= 2:
        xs = list(range(len(tail)))
        mx = sum(xs) / len(xs)
        my = sum(tail) / len(tail)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, tail))
        den = sum((x - mx) ** 2 for x in xs)
        g = num / den if den > 0 else 0.0
    else:
        g = 0.0
    g = max(MC_LEVEL_G_FLOOR, min(MC_LEVEL_G_CLIP, g))
    return L_last, g


def mc_sizes(cand_mus, newcomer, horizons, places, edges, level):
    """Monte Carlo size distributions, one per place (v3).

    cand_mus: list of (mint, [mu per read horizon], sigma) for the candidate
        coins (mu = drift-extrapolated log-cap at that read, sigma = the
        coin's per-mark log-cap volatility).
    newcomer: newcomer_stats() dict (births per mark, mooner fraction).
    horizons: per-read horizons in marks (one per answer read).
    level: (L_last, g) from top_level_forecast(): the top-1 log-cap at the
        last mark and its per-mark growth.
    Returns a sizes list [{place, bin, p}] with per-place sums of 1.

    Each simulated field draws every candidate's cap as its drift path plus
    a random-walk term (dispersion growing with sqrt(horizon) up to a cap,
    so it saturates instead of exploding over 10 hours), plus a Poisson
    number of newborn mooners. A mooner's cap is drawn around the
    trend-extrapolated top level at the read (the level, with noise, minus
    an exponential gap): a newborn that moons lands just below the level
    the market has reached, which is what the rehearsal shows — the #1 cap
    at every mark equals the largest cap among coins born in the last few
    marks. The top 5 of the field are binned; histograms pool over the
    reads and the simulations. The top-5 caps are order statistics of a
    noisy field plus newcomers, so this is right-skewed and newcomer-driven
    where v2's single symmetric normal was not.
    """
    rng = random.Random(MC_SEED)
    n_bins = len(edges) + 1
    n_reads = len(horizons)
    counts = [[0] * n_bins for _ in range(places)]
    births = newcomer["births_per_mark"]
    f_moon = newcomer["f_moon"]
    L_last, g = level
    n_cand = len(cand_mus)
    for j, H in enumerate(horizons):
        sds = []
        for _, mus, sigma in cand_mus:
            sd = sigma * math.sqrt(min(H, MC_VOL_HORIZON_CAP)) if H > 0 else 0.0
            sds.append(min(MC_VOL_SD_CAP, sd))
        lam = max(0.0, births * H * f_moon)
        # top level at this read: trend growth, saturating, with noise.
        L_pred = L_last + max(-0.2, min(g * H, MC_LEVEL_G_MAX)) if H > 0 else L_last
        L_sd = min(MC_LEVEL_SIGMA_CAP,
                   MC_LEVEL_SIGMA_BASE +
                   MC_LEVEL_SIGMA_PER_MARK * math.sqrt(H)) if H > 0 else 0.0
        for _ in range(MC_N_SIM):
            caps = []
            for idx in range(n_cand):
                caps.append(cand_mus[idx][1][j] +
                            sds[idx] * rng.gauss(0.0, 1.0))
            L_j = L_pred + L_sd * rng.gauss(0.0, 1.0) if L_sd > 0 else L_pred
            k_new = _poisson(rng, lam)
            for _ in range(k_new):
                gap = -MC_GAP_MEAN * math.log(max(rng.random(), 1e-12))
                caps.append(L_j - gap)
            caps.sort(reverse=True)
            for k in range(places):
                c = caps[k] if len(caps) > k else (caps[-1] if caps else 1.45)
                counts[k][bisect.bisect_right(edges, c)] += 1
    total = MC_N_SIM * n_reads
    sizes = []
    for k in range(places):
        for b in range(n_bins):
            p = (counts[k][b] + MC_BIN_ALPHA) / (total + MC_BIN_ALPHA * n_bins)
            p = math.floor(p * 1000000) / 1000000.0
            if p > 0:
                sizes.append({"place": k + 1, "bin": b, "p": p})
    return sizes


def build_forecast(marks, places, edges, horizon_marks):
    """Build the forecast.

    horizon_marks is one float (the horizon to the middle answer read, as
    the dev backtest calls it) or a list of per-read horizons in marks (as
    main() passes it); the order model uses the middle horizon and the
    Monte Carlo size model pools over all of them.
    """
    if isinstance(horizon_marks, (int, float)):
        horizons = [float(horizon_marks)]
    else:
        horizons = [float(h) for h in horizon_marks]
    horizon_mid = horizons[len(horizons) // 2]
    last_idx = len(marks) - 1
    feats = coin_features(marks)
    # candidates: coins seen at the last mark, ranked by predicted mu at the
    # middle read. (If the last mark were empty, fall back to every coin.)
    cands = []
    for mint, feat in feats.items():
        if feat["last_idx"] != last_idx:
            continue
        cands.append((predict_mu(feat, last_idx, horizon_mid), mint))
    if not cands:
        for mint, feat in feats.items():
            cands.append((predict_mu(feat, last_idx, horizon_mid), mint))
    cands.sort(reverse=True)
    cands = cands[:N_CANDIDATES]
    rank_of = {mint: i + 1 for i, (_, mint) in enumerate(cands)}
    probabilities, top5 = assemble(cands, rank_of, places, horizon_mid)
    # Monte Carlo sizes: each candidate's drift path at every read, plus the
    # newcomer process.
    cand_mus = []
    for _, mint in cands:
        feat = feats[mint]
        mus = [predict_mu(feat, last_idx, H) for H in horizons]
        cand_mus.append((mint, mus, feat["sigma"]))
    sizes = mc_sizes(cand_mus, newcomer_stats(marks), horizons, places,
                     edges, top_level_forecast(marks))
    return {"probabilities": probabilities, "top5": top5, "sizes": sizes}


def assemble(cands, rank_of, places, horizon_marks):
    """Order probabilities and top5 (unchanged from v2).

    Each place's probabilities are a Gaussian over predicted-rank distance,
    scaled so the named coins take (1 - unnamed) of the place; the rest is
    left unnamed for the coins that move and the coins not yet born.
    Returns (probabilities, top5).
    """
    unnamed = min(UNNAMED_MAX, UNNAMED_BASE + UNNAMED_PER_MARK * horizon_marks)
    spread = min(RANK_SPREAD_MAX,
                 RANK_SPREAD_BASE + RANK_SPREAD_PER_MARK * horizon_marks)
    probabilities = []
    for k in range(1, places + 1):
        weights = []
        for _, mint in cands:
            d = (rank_of[mint] - k) / spread
            weights.append(math.exp(-0.5 * d * d))
        total = sum(weights)
        named = 1.0 - unnamed
        for (_, mint), w in zip(cands, weights):
            p = named * w / total
            if p >= 0.000001:
                probabilities.append({"mint": mint, "place": k,
                                      "p": round(p, 6)})
    top5 = [mint for _, mint in cands[:places]]
    return probabilities, top5


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    marks = load_history(os.path.join(INPUT_DIR, "history.ndjson"))
    if not marks:
        raise SystemExit("the history holds no ok mark")
    places = int(task.get("places", 5))
    edges = [float(e) for e in task["size"]["edges"]]
    cadence = float(task.get("cadence_minutes", 10))
    reads = task["reads"]
    last_at = parse_time(marks[-1]["at"])
    read_times = [parse_time(r) for r in reads]
    horizons = [max(0.0, (rt - last_at) / (cadence * 60.0))
                for rt in read_times]
    forecast = build_forecast(marks, places, edges, horizons)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task.get("output", "forecast.json")), "w", encoding="utf-8") as f:
        f.write(json.dumps(forecast) + "\n")


if __name__ == "__main__":
    main()
