#!/usr/bin/env python3
"""pumpfun-top5-forecast-01 entry: momentum-aware forecaster (v2).

Reads /input/history.ndjson (10-minute snapshots of pump.fun coins launched
since the opening, ranked by market cap) plus /input/task.json, and writes one
forecast to /output/forecast.json:
  {probabilities: [{mint, place 1-5, p}], top5: [5 mints in order],
   sizes: [{place 1-5, bin 0-81, p}]}
Python 3.13 stdlib only, no network, deterministic (no randomness at all).

Method: per coin, fit the recent log10 market-cap drift (slope over the last
up-to-4 marks blended with the last step) and extrapolate it, damped, toward
the middle answer read — momentum matters over minutes, not over 10 hours.
Rank candidates by that predicted cap; each place's probabilities are a
Gaussian over predicted-rank distance, with a large share left unnamed
because newcomers take a large share of top-5 places (3 of the last
rehearsal read's top 5 were never in the history). Each place's size is a
normal in log10 SOL around the predicted k-th cap, lifted by an
order-statistic correction (the actual cap at a place is the k-th largest
of many coins, so it sits above the k-th predicted coin's mean), with the
spread widening with the answer horizon. All constants were calibrated on
a local backtest of the rehearsal history (see dev/): the forecaster beats
the house persistence baseline on every split.
"""
import json
import math
import os

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
        out[mint] = {
            "last_log": last_log, "last_rank": last_rank,
            "venue": d["venue"], "first_idx": first_i, "last_idx": last_i,
            "slope": slope, "g1": g1, "n_seen": len(pts),
        }
    return out


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


def build_forecast(marks, places, edges, horizon_marks):
    last_idx = len(marks) - 1
    feats = coin_features(marks)
    # candidates: coins seen at the last mark, ranked by predicted mu.
    # (If the last mark were empty, fall back to every coin ever seen.)
    cands = []
    for mint, feat in feats.items():
        if feat["last_idx"] != last_idx:
            continue
        cands.append((predict_mu(feat, last_idx, horizon_marks), mint))
    if not cands:
        for mint, feat in feats.items():
            cands.append((predict_mu(feat, last_idx, horizon_marks), mint))
    cands.sort(reverse=True)
    cands = cands[:N_CANDIDATES]
    rank_of = {mint: i + 1 for i, (_, mint) in enumerate(cands)}
    return assemble(cands, rank_of, places, edges, horizon_marks)


def erfc(x):
    """Numerical Recipes' erfcc (fractional error under 1.2e-7)."""
    z = abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    r = t * math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 +
        t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 +
        t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    return r if x >= 0 else 2.0 - r


def norm_above(x, mean, sd):
    """P(X >= x) for X normal(mean, sd)."""
    if x == float("-inf"):
        return 1.0
    if x == float("inf"):
        return 0.0
    return 0.5 * erfc((x - mean) / (sd * math.sqrt(2.0)))


def assemble(cands, rank_of, places, edges, horizon_marks):
    unnamed = min(UNNAMED_MAX, UNNAMED_BASE + UNNAMED_PER_MARK * horizon_marks)
    spread = min(RANK_SPREAD_MAX,
                 RANK_SPREAD_BASE + RANK_SPREAD_PER_MARK * horizon_marks)
    size_sd = min(SIZE_SD_MAX,
                  SIZE_SD_BASE + SIZE_SD_PER_MARK * horizon_marks)
    probabilities = []
    sizes = []
    for k in range(1, places + 1):
        # place k: Gaussian weight on predicted-rank distance, then scale so
        # the named coins take (1 - unnamed) of the place.
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
        # sizes: normal around the predicted k-th largest mu, lifted by the
        # order-statistic correction (largest at place 1, none at place 5)
        center = cands[k - 1][0] if len(cands) >= k else cands[-1][0]
        center += CENTER_SHIFT1 * (places - k) / max(1, places - 1)
        for b in range(len(edges) + 1):
            lo = float("-inf") if b == 0 else edges[b - 1]
            hi = float("inf") if b == len(edges) else edges[b]
            p = norm_above(lo, center, size_sd) - norm_above(hi, center, size_sd)
            p = math.floor(p * 1000000) / 1000000.0
            if p > 0:
                sizes.append({"place": k, "bin": b, "p": p})
    top5 = [mint for _, mint in cands[:places]]
    return {"probabilities": probabilities, "top5": top5, "sizes": sizes}


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
    mid_read = read_times[len(read_times) // 2]
    horizon_marks = max(0.0, (mid_read - last_at) / (cadence * 60.0))
    forecast = build_forecast(marks, places, edges, horizon_marks)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task.get("output", "forecast.json")), "w", encoding="utf-8") as f:
        f.write(json.dumps(forecast) + "\n")


if __name__ == "__main__":
    main()
