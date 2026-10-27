# Methodology and validation

This document explains exactly what the model does, how the numbers are produced, what was tested, and what was not.

## 1. Data
* **Play-by-play:** nflverse (`play_by_play_1999 ... 2026`). **Schedule, results, referees, roof, weather:** nflverse `nfldata/games.csv`.
* **Flags are parsed from the play description text.** `PENALTY on` (upper case) marks an enforced (accepted) penalty; `Penalty on ... declined` / `... offsetting` mark flags that were not enforced.
  *Flags thrown* = enforced + declined + offsetting. *Accepted* = enforced.
* The parser was validated on all 28 seasons: 104,682 flags (through 2026 week 6), 100% matched to a game team. For accepted penalties it matches nflfastR's own `penalty` column to within about 0.05 per game
  (2023: correlation 0.999, 97.4% of games exact; the remainder are plays with two enforced penalties, which nflfastR records once).
* **Parser coverage (two fixes made late):** the 2024+ kickoff penalties ("Kickoff Out of Bounds", "Kickoff Short of Landing Zone") are written as "placed at" rather than "enforced" and were initially missed
  (0.13 per game in 2024, 0.30 in 2025, 0.26 in 2026). The pattern now matches exactly 141 such flags in 2024-2026, all on plays nflfastR also marks as penalties, with no effect before 2024. Three played games
  (1999_01_BAL_STL, 2000_03_SD_KC, 2000_06_BUF_MIA) have no play-by-play rows in nflverse, so they are excluded instead of being counted as 0-flag games. After both fixes, the only accepted penalties the parser does not
  recover are 18 nflfastR rows from 2010-2011 that say "play under review" and carry no penalty type.
* **Not observable anywhere in public data: flags that were thrown and then picked up (waved off).** If a sportsbook counts them in "flags thrown", this model undercounts that market.
* Relocated franchises are normalised (STL to LA, SD to LAC, OAK to LV). Only played regular-season games are used. A game with no listed referee gets its own prior-only crew level
  (so an unknown crew is averaged over properly instead of being treated as one shared "unknown" referee).

## 2. Model
Unit = one team in one game (two rows per game). Hierarchical Bayesian, fit with PyMC (NUTS via nutpie).

```
thrown[g,t]   ~ NegativeBinomial( mu[g,t], alpha )
accepted[g,t] ~ Binomial( thrown[g,t], p[g,t] )

log mu[g,t] = a0
            + season_level[s]                 random walk across seasons (league-wide flag level)
            + week_effect[w]                  random walk across weeks 1-18, shared by all seasons (week 1 = reference)
            + team_effect[t,s]                random walk across seasons per team (initial spread + yearly drift)
            + opponent_effect[o]              the opponent's tendency to induce flags
            + crew_effect[referee]            persistent, partially pooled (shrinks small samples toward zero)
            + X . beta                        home, dome, divisional, primetime, wind/10, cold (<40F outdoors)
            + game_effect[g]                  shared by BOTH teams: captures within-game correlation (~0.22)
logit p[g,t] = c0 + acceptance_team[t] + acceptance_crew[referee]
```
Priors (weakly informative): a0 ~ N(log 7.2, 0.3); scale parameters ~ HalfNormal (season walk 0.10, team 0.10, team drift 0.04, opponent 0.05,
crew 0.08, week 0.05, game 0.15, acceptance 0.15); beta ~ N(0, 0.15); NB dispersion alpha ~ Gamma(3, 0.1). Sampler: NUTS, target_accept 0.9.

Structure was chosen from what the data showed: flag counts are over-dispersed (variance/mean about 1.6 for game totals); the two teams' counts correlate within a game (0.22);
the league level swings by era (13 to 17 flags per game); and flags decline through the season.

## 3. From posterior to prediction
For each game: draw posterior parameters, draw a fresh game effect, sample both teams' flag counts (gamma-Poisson), then thin to accepted counts.
That is 12 simulated games per posterior draw (24,000 per game with the shipped 4-chain fit). All four markets (game / team, thrown / accepted) come from the same simulations,
so they are consistent with each other. Future seasons continue the season-level and team random walks, so predictions further ahead are wider.

## 4. Referee-crew handling
* **Assigned crew:** that crew's effect is applied, **scaled to 60%** (`--crew-shrink`). The scale comes from a backtest slope test (section 6): raw crew adjustments moved
  predictions about twice as far as the data supported (slope 0.56, CI 0.07 to 1.03).
* **Unassigned crew:** the model averages over the 17 active crews (a random active crew per simulation), which widens the line honestly.
* **Crew effect in flags** (`crew_effect_flags` in the CSV, "crew effect" in the sheet) is computed **exactly**, from the same posterior: expected flags with the assigned crew minus expected flags
  averaged over the active crews. It carries no simulation noise. (An earlier version differenced two simulations and was noisy by about +/-0.1 flags.)
* Estimates for crews with little history (for example second-season referees) are shrunk by the model but remain wide; see the 80% interval in `python3 nfl_flags.py crews`.
* The crew is the head referee. The NFL can substitute members of a crew from week to week; that is not modelled.

## 5. Pricing
Main line = the half-point closest to 50/50 of the simulated distribution. For discrete counts the nearest half-point can still be 40-60%, so the remaining skew goes into the odds.
A **7% multiplicative hold** is applied to the fair probabilities (`q = p x 1.07`), odds are rounded to 5-cent steps and never inside (-100, +100). Alternate lines are the neighbouring half-points.
The hold is an assumption for a thin prop market, not a statement about any real book. If probabilities are calibrated, every quoted side then has expected ROI of -hold/(1+hold) = -6.5%
regardless of its probability; this identity is used for validation below.

## 6. Validation, in the order it was done
Backtests are **walk-forward**: refit using only games already played, then predict the next games. Test set: 816 regular-season games, 2023-2025.

| Question | Result |
|---|---|
| Does the model beat naive baselines? | Yes against trailing league/team averages (log score +0.015 to +0.037, CIs above zero) **but this gain is entirely the week-of-season trend.** |
| Does it beat "league average x week-of-season factor"? | **No: a statistical tie** (log-score differences between -0.003 and +0.011, every 95% CI spans zero). |
| Is it calibrated? | Predictive distributions: PIT KS p-values 0.09 to 0.89 across variants. Priced lines: blanket over and under bets lose about the theoretical 6.5%, inside the CI on every market; reliability by probability bin is good. |
| Do crew effects help out-of-sample? | Raw data: crews differ by about +/-0.64 accepted flags per game (permutation p <= 0.003, about 2.4% of variance). Log score: no detectable gain (differences within +/-0.006). Slope test: positive but about half the size the model applies. |
| Does more history help? | Not detectably: 1999+ vs 2018+ differ by -0.008 to 0.000 in log score, CIs span zero. |
| Do play-volume features help? | Expected pass rate has a small real effect (+1.6% per SD); expected total plays has none. Too small to detect on 816 games, so they are not in the model. |
| Variance by penalty type | Crew signal is strongest on offensive holding/blocking (SD 0.34 flags/game), then pre-snap procedural (0.21), personal fouls (0.16), pass-coverage judgment calls (0.11). Team-season effects are mostly pre-snap procedural (SD 0.35). |

**Reproducibility tests of the shipped tool:** the combined tool reproduces the earlier separately-written pipeline exactly (420 of 420 lines); a full independent refit agrees with a prior fit to a mean
0.037 flags per market with identical main lines on 100% of 84 markets.

## 7. What was tuned on the test data (disclosure)
* The week-of-season effect was added after seeing the first version's test diagnostics (the pattern is also visible in the 2018-22 training years; no hyperparameters were tuned on the test set).
* `--crew-shrink 0.6` was set from the slope test on the same backtest it is judged on.
* The 7% hold is an assumption, not an estimate.

## 8. Known weaknesses
1. **Ties a naive baseline.** The model is a reasonable reference price generator, not a source of edge. Model-implied EV against any real price is overconfident.
2. **Training-window sensitivity.** Using 2018+ instead of 1999+ lowers expected flags by about 0.5-0.9 per game; the main line changes on about 38% of markets for a typical week.
3. **Week-of-season shape.** The pooled curve is flatter mid-season than recent seasons (about -5% by week 12 vs roughly -10% recently). Backtest bias in weeks 10-12 was +0.6 flags (1.4 standard errors).
   Before any in-season data exists, early weeks are under-predicted when a season runs hot or cold (-0.8 to -1.0 in weeks 1-6 of the backtest).
4. Divisional and dome coefficients are small and sign-unstable across fits; the home effect (-6%) is robust. Weather defaults to neutral.
5. No real sportsbook penalty prop prices were available, so the lines have never been compared with an actual market.
6. Sampler diagnostics are acceptable but not pristine for the league-level hyperparameters (see the README for the final fit's numbers).
