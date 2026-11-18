"""gta6-twitch-ccu-forecast-01 entry: analog (comparables) quantile forecaster.

Model: for each forecast hour h, the distribution of log(viewers_h / viewers_1) is estimated
from the case's own `comparables` — past launches whose full release days ship inside the
case file. Each comparable is weighted by a Gaussian kernel over case features (hour-1 level,
how far hour 1 already sits above the pre-release baseline, and the launch's unlock pattern
parsed from its text), and the hour's five quantiles blend the weighted empirical quantiles
of the comparables' log-ratios with a weighted normal fit. With too little comparable data
it falls back to the kit starter's persistence forecast.

Run by the grader as `python3 -P -s /bot/main.py` (Python 3.13, standard library only).
Outside the grader, MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR stand in for /input and /output.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a sibling module would import.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard-normal quantiles at LEVELS (the starter's digits).
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
SIGMA0 = 0.15  # the persistence baseline's spread per sqrt hour

# Tuned by leave-one-out over the 37 comparable launches (see README.md).
PARAMS = {
    "s_logp1": 0.42,     # kernel scale for log10(hour-1 viewers)
    "s_jump4": 1.2,      # kernel scale for log(p1 / pre-release 4h mean)
    "s_roll": 0.65,      # kernel scale for the rolling-midnight flag
    "s_glob": 0.65,      # kernel scale for the simultaneous-global flag
    "s_early": 0.90,     # kernel scale for the early-access flag
    "s_uh": 5.0,         # kernel scale for the parsed unlock hour (UTC)
    "alpha": 0.40,       # weight on empirical weighted quantiles vs the normal fit
    "sigma_floor": 0.12, # smallest log spread any hour may take
    "min_pool": 5,       # below this many usable comparables, use persistence
}


def round2(x: float):
    """x rounded to 2 decimals, half up (the starter's rounding; whole numbers stay ints)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


TIME_RE = re.compile(r"(\d{1,2}):(\d{2})\s*Z\b")


def unlock_flags(text):
    """Unlock-pattern features from a launch's notes/unlock text.

    Returns (rolling, global, early, unlock_hour): unlock_hour is a representative UTC
    hour of the main unlock (-6 for a rolling-midnight launch, already live before the
    UTC day begins), or None when the text gives no usable time."""
    t = (text or "").lower()
    rolling = ("midnight local" in t) or ("rolling" in t) or ("midnight in each" in t) \
        or ("midnight on" in t and "local" in t)
    glob = ("simultaneous" in t) or ("global unlock" in t) or ("single global" in t) \
        or ("one global" in t)
    early = "early access" in t
    times = sorted(int(a) + int(b) / 60.0 for a, b in TIME_RE.findall(text or ""))
    if rolling:
        uh = -6.0
    elif times:
        uh = times[len(times) // 2]
    else:
        uh = None
    return (1.0 if rolling else 0.0), (1.0 if glob else 0.0), (1.0 if early else 0.0), uh


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def case_features(day_start_ts, p1, pre_points, text):
    """Features of one launch. pre_points: [(ts, viewers)] before day_start."""
    p1c = max(1.0, float(p1))
    pre24 = [v for (t, v) in pre_points if day_start_ts - 86400 <= t < day_start_ts]
    pre4 = [v for (t, v) in pre_points if day_start_ts - 14400 <= t < day_start_ts]
    pre13 = [v for (t, v) in pre_points if day_start_ts - 13 * 3600 <= t < day_start_ts]
    pre13far = [v for (t, v) in pre_points if day_start_ts - 13 * 3600 <= t < day_start_ts - 3 * 3600]
    pre3 = [v for (t, v) in pre_points if day_start_ts - 3 * 3600 <= t < day_start_ts]
    preall = [v for (t, v) in pre_points if day_start_ts - 14 * 86400 <= t < day_start_ts]
    m24, m4, mall = _mean(pre24), _mean(pre4), _mean(preall)
    m13, m13far, m3 = _mean(pre13), _mean(pre13far), _mean(pre3)
    roll, glob, early, uh = unlock_flags(text)
    return {
        "logp1": math.log10(p1c),
        "jump24": math.log(p1c / max(1.0, m24)) if m24 is not None else 0.0,
        "jump4": math.log(p1c / max(1.0, m4)) if m4 is not None else 0.0,
        "jump13": math.log(p1c / max(1.0, m13)) if m13 is not None else 0.0,
        "ramp": math.log(max(1.0, m3) / max(1.0, m13far)) if (m3 is not None and m13far is not None) else 0.0,
        "prelevel": math.log(p1c / max(1.0, mall)) if mall is not None else 0.0,
        "roll": roll, "glob": glob, "early": early, "uh": uh,
    }


def pool_entry(comp):
    """One comparable -> {features, logratio per hour, prior} or None if its day is unusable."""
    ds = parse_ts(comp.get("day_start"))
    if ds is None:
        return None
    by_hour = {}
    pre = []
    for pt in comp.get("hourly") or []:
        t = parse_ts(pt.get("from"))
        if t is None:
            continue
        v = pt.get("viewers")
        if v is None:
            continue
        h = int((t - ds) // 3600)
        if 0 <= h < 24:
            by_hour.setdefault(h, []).append(float(v))
        elif -14 * 24 <= h < 0:
            pre.append((t, float(v)))
    if 0 not in by_hour:
        return None
    day = {h: _mean(vs) for h, vs in by_hour.items()}
    p1 = day[0]
    if p1 is None or p1 <= 0:
        return None
    have = sum(1 for h in range(1, 24) if h in day)
    if have < 12:
        return None
    feats = case_features(ds, p1, pre, comp.get("unlock") or "")
    logratio = {h + 1: math.log(max(1.0, day[h]) / p1) for h in range(1, 24) if h in day}
    return {"features": feats, "logratio": logratio, "prior": 1.0}


def kernel_weight(tf, cf):
    """Gaussian kernel over feature differences between target and comparable."""
    d2 = 0.0
    d2 += ((tf["logp1"] - cf["logp1"]) / PARAMS["s_logp1"]) ** 2
    d2 += ((tf["jump4"] - cf["jump4"]) / PARAMS["s_jump4"]) ** 2
    d2 += ((tf["roll"] - cf["roll"]) / PARAMS["s_roll"]) ** 2
    d2 += ((tf["glob"] - cf["glob"]) / PARAMS["s_glob"]) ** 2
    d2 += ((tf["early"] - cf["early"]) / PARAMS["s_early"]) ** 2
    if tf.get("uh") is not None and cf.get("uh") is not None:
        d2 += ((tf["uh"] - cf["uh"]) / PARAMS["s_uh"]) ** 2
    return math.exp(-0.5 * d2)


def weighted_quantile(pairs, tau):
    """pairs: [(value, weight)] -> the tau weighted quantile (linear on cumulative weight)."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return pairs[len(pairs) // 2][0]
    target = tau * total
    cum = 0.0
    prev_v, prev_c = pairs[0][0], 0.0
    for v, w in pairs:
        cum += w
        if cum >= target:
            if cum == prev_c:
                return v
            frac = (target - prev_c) / (cum - prev_c)
            return prev_v + frac * (v - prev_v)
        prev_v, prev_c = v, cum
    return pairs[-1][0]


def hour_quantiles_log(pairs):
    """The five log-quantiles for one hour from [(logratio, weight)] pairs."""
    total = sum(w for _, w in pairs)
    mu = sum(v * w for v, w in pairs) / total
    var = sum(w * (v - mu) ** 2 for v, w in pairs) / total
    n_eff = total * total / sum(w * w for _, w in pairs)
    if n_eff > 1.5:
        var *= n_eff / (n_eff - 1.0)
    sigma = max(math.sqrt(max(var, 0.0)), PARAMS["sigma_floor"])
    alpha = PARAMS["alpha"]
    out = []
    for tau, z in zip(LEVELS, Z):
        emp = weighted_quantile(pairs, tau)
        fit = mu + z * sigma
        out.append(alpha * emp + (1.0 - alpha) * fit)
    out.sort()
    return out


def persistence_quantiles(p1, h):
    return [round2(p1 * math.exp(z * SIGMA0 * math.sqrt(h - 1))) for z in Z]


def forecast_case(case):
    known = case.get("known") or []
    first = next((k for k in known if k.get("hour") == 1), None)
    p1 = None
    if first is not None and first.get("viewers") is not None:
        p1 = float(first["viewers"])
    if p1 is None:
        hist = case.get("history_hourly") or []
        if hist and hist[-1].get("viewers") is not None:
            p1 = float(hist[-1]["viewers"])
    if p1 is None or p1 <= 0:
        p1 = 1.0
    p1 = max(1.0, p1)

    hours = [row.get("hour") for row in (case.get("forecast") or [])]
    hours = [h for h in hours if isinstance(h, int)] or list(range(2, 25))

    ds = parse_ts(case.get("day_start"))
    pool = []
    if ds is not None:
        for comp in case.get("comparables") or []:
            e = pool_entry(comp)
            if e is not None:
                pool.append(e)

    rows = []
    if len(pool) < PARAMS["min_pool"] or ds is None:
        for h in hours:
            rows.append({"hour": h, "q": persistence_quantiles(p1, h)})
        return {"hours": rows}

    pre = []
    for pt in case.get("history_hourly") or []:
        t = parse_ts(pt.get("from"))
        if t is not None and pt.get("viewers") is not None and t < ds:
            pre.append((t, float(pt["viewers"])))
    tf = case_features(ds, p1, pre, case.get("notes") or "")
    weights = [e["prior"] * kernel_weight(tf, e["features"]) for e in pool]

    for h in hours:
        pairs = [(e["logratio"][h], w) for e, w in zip(pool, weights) if h in e["logratio"] and w > 0]
        if len(pairs) < PARAMS["min_pool"]:
            rows.append({"hour": h, "q": persistence_quantiles(p1, h)})
            continue
        logq = hour_quantiles_log(pairs)
        q = []
        for lq in logq:
            v = p1 * math.exp(lq)
            v = min(max(v, 0.01), 1e9)
            q.append(round2(v))
        q = sorted(q)
        rows.append({"hour": h, "q": q})
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            # Never leave a case unwritten: persistence around 1 is the least-bad fallback.
            forecast = {"hours": [{"hour": h, "q": persistence_quantiles(1.0, h)} for h in range(2, 25)]}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
