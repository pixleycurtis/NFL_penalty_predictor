# NFL penalty-flag toolkit

Predict the number of flags in any NFL game, see how the assigned referee crew changes it, and generate sportsbook-style prop lines (flags thrown and penalties accepted, for the game and each team).
Everything runs locally from one script, `nfl_flags.py`, using the trained model shipped in `model/`.

**Read this first:** in backtests the model is statistically indistinguishable from a naive "league average x week-of-season" baseline. Treat its lines as reasonable reference prices, **not edges**.
Full limitations are in section 7 and `METHODOLOGY.md`.

## 1. Install
**To run the tool** (`predict`, `week`, `crews`) you only need:
```bash
python3 -m pip install -r requirements.txt        # numpy, pandas, certifi, pyarrow (pyarrow is only needed by `actual`)
```
You do **not** need PyMC or anything else unless you want to refit the model with `update`. The shipped model in `model/` is ready to use.

**To refit** (`update`) pick ONE route:
* **A. Recommended: a Python 3.11-3.13 environment, ideally 3.12** (the combination this was developed and timed on; fastest). **Not Python 3.14.**
  ```bash
  python3.12 -m venv .venv && source .venv/bin/activate
  pip install -r requirements-refit.txt
  ```
  No Python 3.12? Install it from python.org (then run its `Install Certificates.command`), or let `uv` fetch it without admin rights (not tested on a Mac):
  ```bash
  python3 -m pip install uv
  uv venv --python 3.12 .venv && source .venv/bin/activate
  uv pip install -r requirements-refit.txt
  ```
  `requirements-refit.txt` pins `numba<0.63` and `llvmlite<0.46`: the newest versions have no prebuilt wheels for Intel Macs (see Troubleshooting).
* **B. Stay on your current Python (including 3.14) and avoid numba/llvmlite** (slower; resolves with prebuilt wheels for Intel macOS on Python 3.12-3.14; run end to end on Python 3.12 and 3.14 on Linux, not on macOS)
  ```bash
  python3 -m pip install -r requirements-refit-nonumba.txt
  xcode-select --install                      # PyMC's built-in sampler needs a C compiler
  python3 nfl_flags.py update --since 2018 --chains 2
  ```

## 2. Quick start (no refit needed)
```bash
# this week's slate, with the referee crews, as CSV + a readable sheet
python3 nfl_flags.py week --week 6 --referees-file examples/week6_referees.csv --out week6_lines.csv --md week6_lines.md

# one game, scheduled or hypothetical
python3 nfl_flags.py predict --game-id 2026_06_SEA_DEN --referee "Brad Rogers" --alts 2
python3 nfl_flags.py predict --home KC --away BUF --week 8 --referee "Clay Martin" --temp 38 --wind 15
python3 nfl_flags.py predict --home DAL --away PHI --week 12 --line 16.5 --odds -110/-110     # price a book's line

# how much each crew moves the flag rate
python3 nfl_flags.py crews

# what actually happened (games already played), and how your lines did
python3 nfl_flags.py actual --game-id 2026_06_SEA_DEN --list
python3 nfl_flags.py actual --week 6 --lines week6_lines.csv
```
`week` and `predict --game-id` refresh the schedule from the internet; add `--offline` to use the cached `data/games.csv`. `predict --home/--away/--week --offline` needs no network at all.

## 3. Commands
| command | what it does | key options |
|---|---|---|
| `predict` | Forecast and price one game: expected flags thrown / accepted (game and each team) with intervals, crew effect, main + alternate lines | `--game-id` or `--home --away --week [--season]`; `--referee "Name"`; `--temp --wind --dome --outdoors --primetime`; `--line --odds --market` to price a book line; `--alts N`; `--json` |
| `week` | Price every unplayed game of a week | `--week N [--season]`; `--referees-file crews.csv` (columns `game_id,referee`) or `--referee GAME_ID="Name"`; `--out lines.csv`; `--md sheet.md`; `--alts N` |
| `crews` | Table of referee-crew effects on the flag rate (raw effect with 80% interval; effect vs an average crew; approximate flags per game) | `--all` (include inactive), `--csv file`, `--crew-shrink` |
| `actual` | **What actually happened** in games already played: flags thrown / accepted / declined / offsetting for each team, every flag with quarter and penalty type, and (optionally) your lines settled OVER/UNDER | `--game-id`, or `--home --away --week`, or just `--week N` for a whole slate; `--lines lines.csv` (a `week` output) to settle; `--list` (every flag); `--alts`; `--show-lines`; `--refresh` (re-download this season first); `--offline` |
| `update` | Download the latest nflverse data, rebuild, **refit (about 13-30 minutes)** and overwrite `model/` | `--chains 4 --draws 500 --tune 750` (the shipped settings), `--since 2018` (about 3 minutes), `--model-dir`, `--data-dir`, `--refresh-all` |

Shared pricing options: `--hold 0.07` (margin on the two-way price), `--crew-shrink 0.6` (scale on crew effects), `--nrep 12` (simulations per posterior draw), `--seed`, `--model-dir model_since2018` (alternate model).

## 4. Weekly routine
1. **After a week's games finish:** `python3 nfl_flags.py update --chains 4 --draws 500 --tune 750` (data refreshes automatically; only the current season is re-downloaded).
   The tool warns when the model is older than the latest played games.
2. **When crews are announced** (usually Tuesday or Wednesday; Football Zebras posts "Week N referee assignments"): put them in a CSV
   (`game_id,referee`; copy `examples/week6_referees.csv` as a template; game ids look like `2026_07_KC_BUF`, home team last) and run `week --week N --referees-file ...`.
3. Games without a listed crew are priced over the average of the 17 active crews, which gives wider lines. Names are fuzzy-matched; unknown names are treated as unassigned with a notice.

## 5. Output columns (`--out` CSV)
`game_id, kickoff_et, matchup (AWAY @ HOME), referee, market (game_thrown | game_accepted | team_thrown | team_accepted), team (blank for game markets), kind (main | alt), line,
over_odds, under_odds (American), crew_effect_flags (this crew vs an average active crew, in expected flags; blank if unassigned), model_mean, model_median, p_over_fair, hold_realised, book (always MODEL)`.
Lines are half-points (no pushes). `p_over_fair` is the model's probability before the hold.

## 6. How the lines are built (one paragraph; details in METHODOLOGY.md)
A hierarchical negative-binomial model trained on every regular-season game since 1999 gives full probability distributions for each team's flags. The main line is the half-point closest to 50/50, the rest of the skew goes into the odds, a 7% hold
is applied, and odds are rounded to 5 cents. Crew effects are scaled to 60% because the backtest showed the raw effects were about twice too large. Defaults for hypothetical games: divisional status from the 8 divisions,
dome from each team's recent home roof, mild weather (60F, typical wind), primetime off.

## 6b. Checking what actually happened
`actual` counts the flags in a finished game with the **same method the model is trained on**, so it is the right yardstick for these lines. It reads that season's play-by-play (`data/play_by_play_YYYY.parquet`, downloaded on demand; use `--refresh` right after a game ends, because the file only contains games up to its last download).
* *accepted* = enforced penalties (what box scores call "penalties"); *thrown* = accepted + declined + offsetting (each offsetting flag counts once per team).
* It prints a cross-check against nflfastR's own accepted-penalty count. A difference of 1-2 usually means a play with two enforced penalties.
* **Flags that were thrown and picked up are not in any public data**, so they are not counted. If you settle against a book that counts them, or against the NFL gamebook (which lists every flag, including declined and offsetting), your total can be higher.
* Other places to look: the NFL gamebook PDF (every flag), and box scores on Pro-Football-Reference / ESPN / NFL.com (accepted penalties only).

## 7. Limitations
1. **It only ties a naive baseline** (816 held-out games, 2023-25). Crew effects are real (about +/-0.64 flags per game) but too small to measure out-of-sample. Model EV figures are overconfident.
2. **Training-window sensitivity:** training on 2018+ (`--model-dir model_since2018`) instead of 1999+ lowers expected flags about 0.5-0.9 per game; main lines change on about 38% of markets.
3. **Late-season shape:** the pooled week-of-season curve is probably too flat for recent seasons (weeks 10-12 ran +0.6 flags high in backtests, 1.4 standard errors).
4. **"Flags thrown" excludes flags that were picked up** (not recorded in public data).
5. Divisional and dome effects are small and unstable; weather defaults to neutral. The crew is the head referee; substitutions within a crew are not modelled.
6. The 7% hold is an assumption. **No real sportsbook penalty prop prices were available, so nothing has been compared with an actual market.**
7. Two data fixes were made after an earlier version was shipped: (a) the 2024+ kickoff penalties ("Kickoff Out of Bounds", "Kickoff Short of Landing Zone"; text says "placed at", not "enforced") were not being counted,
   which understated accepted penalties by about 0.13 per game in 2024, 0.30 in 2025 and 0.26 in 2026 (about 2%); (b) three played games from 1999-2000 have no play-by-play at all in nflverse and had been counted as 0-flag games, so they are now excluded.
   Lines produced by older versions of this toolkit run about 0.25 flags per game low against official counts for the same reason. After the fix, parsed accepted counts match nflfastR in 97.5-100% of games per season.
8. Final-fit diagnostics: 4-chain fit (500 draws after 750 tuning, 1999+ data) on the corrected data through 2026 week 6: 0 divergences, max R-hat 1.031, minimum bulk effective sample size 108 (league-level hyperparameters). Alternate 2018+ model: 0 divergences, max R-hat 1.027, minimum ESS 136. Relative to the previous (pre-fix) model, expected flags per game rose by about 0.27 (thrown) and 0.25 (accepted), as the kickoff-penalty correction implies; crew effects moved by 0.5 percentage points or less.

## 8. Troubleshooting
* **`CERTIFICATE_VERIFY_FAILED` (macOS):** the tool tries your system certificates, then `certifi`. Fix: `python3 -m pip install certifi`, or run `Install Certificates.command` in /Applications/Python 3.x,
  or `export SSL_CERT_FILE=$(python3 -c 'import certifi; print(certifi.where())')`. Behind a corporate proxy, point `SSL_CERT_FILE` at your company CA bundle.
* **`Failed building wheel for llvmlite` during `pip install`:** pip found no prebuilt `llvmlite` and tried to compile it from source, which needs LLVM. Two causes were identified:
  (1) the **newest** llvmlite/numba (pulled in by recent PyTensor/ArviZ and by nutpie) no longer ship wheels for **Intel Macs**; the last versions that do are llvmlite 0.45 / numba 0.62.
  (2) **Python 3.14** has no llvmlite wheel for Intel Macs at all. Fix: route A in section 1 (Python 3.11-3.13 plus the pins in `requirements-refit.txt`), or route B (no numba, works on 3.14).
  Removing nutpie alone does **not** avoid it. Checked against PyPI for Intel macOS (macOS 13 wheels): the pinned route-A set resolves on Python 3.11, 3.12 and 3.13 and not on 3.14; the route-B set resolves on 3.12-3.14.
  If it still fails, send `python3 --version`, `uname -m` and `sw_vers`.
* **`No module named 'pymc'` on `update`:** `update` checks this before downloading and prints both install routes. `predict`, `week` and `crews` never need PyMC.
* **Refit speed:** with nutpie, a 4-chain 1999+ refit took about 22 minutes on one CPU core (a 2018+ refit about 4 minutes). PyMC's built-in sampler was about 2.7x slower on a short test; a full-length run without nutpie has not been timed, so start with `--since 2018 --chains 2`.
* **"model trained through ... newer games have been played":** run `update`.
* **Unknown team code:** use nflverse codes (`LA`, `LAC`, `LV`, `JAX`, `WAS`, ...); the error lists the valid ones.
* **Predictions for a game that has been played** print an in-sample warning (the model was trained on it).

## 9. Folder layout
```
nfl_flags.py          the whole tool (data, model, simulation, pricing, CLI)
requirements*.txt      run-only packages, refit packages, and a refit route without numba
README.md             this file
METHODOLOGY.md        model, validation, and every known weakness
model/                shipped posterior + metadata (1999+, trained through the date in meta.json)  <- default
model_since2018/      alternate posterior trained on 2018+ only
examples/             Week 6 2026 lines (csv + readable sheet), the verified crew assignments, crew-effect table
research/             all backtests, ablations, validation scripts and result files (NOT needed to run the tool)
data/                 schedule snapshot (games.csv, 2026-10-19) so `--offline` works immediately; `week`/`update` refresh it, and `update` adds ~600 MB of play-by-play
```
