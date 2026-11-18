[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ngadukun/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ngadukun/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# gta6-twitch-ccu-forecast-01 — entry

`main.py` (Python 3.13, standard library only) reads each `/input/<id>.in` case and writes
`/output/<id>.out`: for hours 2–24 of the launch day, five quantiles (0.05/0.25/0.5/0.75/0.95)
of the hour's live Twitch viewers in the game's category.

## Model

Analog quantile forecasting over the case's own `comparables` (past launches whose full
release days ship inside every case file):

- For each comparable, the release-day log-ratio curve `log(viewers_h / viewers_1)`, h = 2..24.
- Each comparable is weighted by a Gaussian kernel over case features: log10 of the hour-1
  count, log(hour-1 / mean of the last 4 pre-release hours), rolling-midnight / simultaneous-
  global / early-access flags and a representative unlock hour parsed from the launch text.
- Each hour's quantiles blend the weighted empirical quantiles of the neighbours' log-ratios
  (weight 0.40) with a weighted normal fit (bias-corrected weighted mean/sd, sd floored at 0.12).
- If fewer than 5 comparables are usable, it falls back to the kit starter's persistence
  forecast; any per-file failure falls back to a persistence write rather than no output.

## Validation (leave-one-out, exact contest loss)

The 37 usable comparable launches were turned into pseudo-cases (hour 1 known, the other
36–41 launches as the pool) and scored with the contest's exact mean pinball loss on the
log scale, `2·max(τ·d, (τ−1)·d)`, `d = ln y − ln q`:

| forecaster | LOO loss (37 launches) | LOO loss (12 rolling, big launches) |
|---|---|---|
| this entry | **0.384** | **0.250** |
| kit starter (persistence) | 0.930 | 0.357 |
| flat (all quantiles = hour 1) | 1.142 | — |
| pooled average shape, unconditioned | 0.714 | 0.368 |

Outputs verified byte-identical across two runs on the kit's practice inputs, and the
output contract (23 rows, hours 2–24, five finite non-decreasing quantiles in (0, 1e9],
≤ 65,536 bytes) checked on every practice case plus malformed/empty/missing-data cases.
