#!/usr/bin/env python3
"""
nfl_flags.py - one-file NFL penalty-flag model: predict flags in ANY game and price realistic prop lines.

  python3 nfl_flags.py predict --home KC --away BUF --week 8 --referee "Clay Martin" --temp 38 --wind 15
  python3 nfl_flags.py predict --game-id 2026_06_SEA_DEN --alts 2
  python3 nfl_flags.py predict --home DAL --away PHI --week 12 --line 16.5 --odds -110/-110
  python3 nfl_flags.py week --week 6 --out week6.csv [--referee 2026_06_SEA_DEN="Alan Eck"]
  python3 nfl_flags.py update            # download latest nflverse data, rebuild, REFIT (~15 min), save posterior

`predict` and `week` need only numpy + pandas and the shipped ./model/ posterior. `update` also needs pymc, nutpie,
arviz, pyarrow. See README.md for what the model can and cannot do (it ties a naive league-average-by-week baseline in backtests).

What it predicts: expected flags THROWN (accepted + declined + offsetting; picked-up flags are not observable in the data) and
penalties ACCEPTED, for the game and each team, as full probability distributions, plus sportsbook-style lines and odds.
"""
import argparse, difflib, importlib.util, json, math, os, re, ssl, sys, time, urllib.error, urllib.request, warnings
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL_DIR, DATA_DIR = HERE / "model", HERE / "data"
FIRST_SEASON = 1999
COVS = ["home", "dome", "div_game", "primetime", "wind10", "cold"]          # order matters (posterior 'beta')
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{y}.parquet"
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
TEAM_MAP = {"OAK": "LV", "STL": "LA", "SD": "LAC", "LAR": "LA", "LVR": "LV", "ARZ": "ARI", "JAC": "JAX", "WSH": "WAS"}
DIVISIONS = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"], "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"], "AFC West": ["DEN", "KC", "LAC", "LV"],
    "NFC East": ["DAL", "NYG", "PHI", "WAS"], "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"], "NFC West": ["ARI", "LA", "SEA", "SF"]}
TEAM_DIV = {t: d for d, ts in DIVISIONS.items() for t in ts}
FIXED_ROOF = {"ATL", "IND", "HOU", "DAL", "ARI", "MIN", "DET", "NO", "LV", "LA", "LAC"}   # fallback only
MARKET_NAMES = {"game_thrown": "Flags thrown (game)", "game_accepted": "Accepted (game)",
                "team_thrown": "Flags thrown", "team_accepted": "Accepted"}
DISCLAIMER = ("Model-generated reference prices, not sportsbook data. 'Flags thrown' excludes flags that were picked up. "
              "Backtests show the model ties a naive league-average-by-week baseline: do not treat these as edges.")


# =====================================================================================================================
# 1. DATA: nflverse download + flag parsing
# =====================================================================================================================
PEN_RE = re.compile(
    r"(?P<case>PENALTY|Penalty) on (?P<team>[A-Z]{2,3})(?:-[^,]+)?,\s*(?P<ptype>[^,]+?),\s*"
    r"(?:(?:\d+|no)\s+yards?,\s*)?(?P<status>enforced|declined|offsetting|placed at)")
# 'PENALTY on' (upper case) = enforced; 'Penalty on ... declined/offsetting' = not enforced. Case matters.


SSL_HELP = (
    "\nSSL certificate verification failed. This is a common Python-on-macOS problem (python.org installs ship without root certificates).\n"
    "Fix it with ONE of:\n"
    "  1. pip install certifi        (this tool then uses certifi's certificate bundle automatically)\n"
    "  2. open Finder -> Applications -> Python 3.x -> double-click 'Install Certificates.command'\n"
    "  3. export SSL_CERT_FILE=$(python3 -c 'import certifi; print(certifi.where())')   then rerun\n"
    "Then run the command again.")


def _ssl_contexts():
    """System certificates first (honours SSL_CERT_FILE and corporate proxies); certifi's bundle as a fallback (fixes python.org macOS)."""
    ctxs = [ssl.create_default_context()]
    try:
        import certifi
        ctxs.append(ssl.create_default_context(cafile=certifi.where()))
    except Exception:
        pass
    return ctxs


def _is_ssl_error(e):
    return (isinstance(e, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(e)
            or (isinstance(e, urllib.error.URLError) and isinstance(getattr(e, "reason", None), ssl.SSLCertVerificationError)))


def _download(url, dest: Path, retries=3):
    dest.parent.mkdir(parents=True, exist_ok=True)
    err = None
    for ctx in _ssl_contexts():
        for k in range(retries):
            try:
                tmp = dest.with_suffix(dest.suffix + ".part")
                with urllib.request.urlopen(url, timeout=180, context=ctx) as r, open(tmp, "wb") as f:
                    f.write(r.read())
                os.replace(tmp, dest)
                return
            except Exception as e:
                err = e
                if _is_ssl_error(e):
                    break                               # retrying cannot fix a certificate problem; try the next certificate source
                time.sleep(3)
        else:
            continue
        if _is_ssl_error(err):
            continue
    raise RuntimeError(f"download failed: {url} ({err})" + (SSL_HELP if _is_ssl_error(err) else ""))


def load_schedule(data_dir=DATA_DIR, refresh=True, quiet=False):
    """nfldata games.csv (schedule, results, referees, roof, weather). Falls back to a cached copy when offline."""
    path = Path(data_dir) / "games.csv"
    if refresh:
        try:
            _download(GAMES_URL, path)
        except Exception as e:
            if not quiet:
                print(f"[schedule] could not refresh ({e}); using cached copy" if path.exists() else f"[schedule] unavailable ({e})")
    if not path.exists():
        return None
    g = pd.read_csv(path)
    g[["home_team", "away_team"]] = g[["home_team", "away_team"]].replace(TEAM_MAP)
    return g


def _status(x):
    """'placed at' is how the 2024+ kickoff penalties (out of bounds, short of the landing zone) are written; they are accepted penalties."""
    return "enforced" if x == "placed at" else x


def parse_flags(pbp_path):
    d = pd.read_parquet(pbp_path, columns=["game_id", "play_id", "desc"])
    d = d[d.desc.notna() & d.desc.str.contains("penalty on", case=False, regex=False)]
    rows = [(gid, pid, m.group("team"), _status(m.group("status")))
            for gid, pid, desc in zip(d.game_id, d.play_id, d.desc) for m in PEN_RE.finditer(desc)]
    return pd.DataFrame(rows, columns=["game_id", "play_id", "team", "status"])


def build_team_games(data_dir=DATA_DIR, since=FIRST_SEASON):
    """Team-game table (2 rows per PLAYED regular-season game, 1999+): flags thrown / accepted + context."""
    games = load_schedule(data_dir, refresh=False)
    games = games[(games.game_type == "REG") & games.home_score.notna() & (games.season >= since)].copy()
    seasons = sorted(games.season.unique())
    pens, have = [], set()
    for s in seasons:
        p = Path(data_dir) / f"play_by_play_{s}.parquet"
        if not p.exists():
            raise SystemExit(f"missing {p}; run `update` to download")
        have |= set(pd.read_parquet(p, columns=["game_id"]).game_id.unique())
        x = parse_flags(p)
        x["team"] = x.team.replace(TEAM_MAP)
        pens.append(x[x.game_id.isin(games.game_id)])
    gaps = games[~games.game_id.isin(have)]
    if len(gaps):      # nflverse has no play-by-play at all for a few 1999-2000 games; counting them would insert false zeros
        print(f"[data] excluding {len(gaps)} played game(s) with NO play-by-play (a data gap, not a penalty-free game): {', '.join(gaps.game_id.head(8))}")
        games = games[games.game_id.isin(have)].copy()
    pens = pd.concat(pens, ignore_index=True).drop_duplicates(["game_id", "play_id", "team", "status"])
    cnt = (pens.assign(n=1).pivot_table(index=["game_id", "team"], columns="status", values="n", aggfunc="sum", fill_value=0).reset_index())
    for c in ["enforced", "declined", "offsetting"]:
        if c not in cnt:
            cnt[c] = 0
    h = games[["game_id", "season", "week", "home_team", "away_team"]]
    tg = pd.concat([h.rename(columns={"home_team": "team", "away_team": "opp"}).assign(home=1),
                    h.rename(columns={"away_team": "team", "home_team": "opp"}).assign(home=0)], ignore_index=True)
    tg = tg[["game_id", "season", "week", "team", "opp", "home"]].merge(cnt, on=["game_id", "team"], how="left")
    for c in ["enforced", "declined", "offsetting"]:
        tg[c] = tg[c].fillna(0).astype(int)
    matched = pens.merge(tg[["game_id", "team"]].drop_duplicates(), on=["game_id", "team"]).shape[0]
    print(f"[data] {len(pens):,} flags parsed; {matched / len(pens):.4%} matched to a game team")
    tg["accepted"] = tg.enforced
    tg["thrown"] = tg.enforced + tg.declined + tg.offsetting
    ctx = games[["game_id", "referee", "roof", "temp", "wind", "div_game", "gametime"]].copy()
    ctx["dome"] = ctx.roof.isin(["dome", "closed"]).astype(int)
    ctx["temp"] = ctx.temp.where(ctx.dome == 0, 70.0).fillna(ctx.temp.median())
    ctx["wind"] = ctx.wind.where(ctx.dome == 0, 0.0).fillna(0.0)
    ctx["wind10"] = ctx.wind / 10.0
    ctx["cold"] = ((ctx.temp < 40) & (ctx.dome == 0)).astype(int)
    ctx["primetime"] = (pd.to_datetime(ctx.gametime, format="%H:%M", errors="coerce").dt.hour >= 19).astype(int)
    ctx["div_game"] = ctx.div_game.fillna(0).astype(int)
    tg = tg.merge(ctx.drop(columns=["gametime", "roof", "temp", "wind"]), on="game_id", how="left")
    # unknown crew -> its own prior-only level per game (properly marginalised), not one shared pseudo-crew
    tg["referee"] = np.where(tg.referee.isna(), "UNK_" + tg.game_id, tg.referee)
    return tg.sort_values(["season", "week", "game_id", "home"], ascending=[True, True, True, False]).reset_index(drop=True), games


# =====================================================================================================================
# 2. MODEL: hierarchical negative binomial (fit once, then simulate)
# =====================================================================================================================
def fit_model(tg, draws=500, tune=500, chains=2, seed=2026, cores=None):
    """
    thrown[g,t]   ~ NegBin(mu, alpha);   accepted[g,t] ~ Binomial(thrown, p)
    log mu = a0 + season level (random walk) + week-of-season (random walk) + team-season effect (random walk)
           + opponent + referee crew + X@beta [home, dome, divisional, primetime, wind, cold] + game effect shared by both teams
    logit p = c0 + team + crew
    """
    import pymc as pm
    import pytensor.tensor as pt
    teams, refs = sorted(tg.team.unique()), sorted(tg.referee.unique())
    first = int(tg.season.min())
    tix, rix = {t: i for i, t in enumerate(teams)}, {r: i for i, r in enumerate(refs)}
    t_i, o_i = tg.team.map(tix).values, tg.opp.map(tix).values
    r_i, s_i = tg.referee.map(rix).values, tg.season.values - first
    w_i, g_i = (tg.week.values - 1).astype(int), pd.factorize(tg.game_id)[0]
    X, y, acc = tg[COVS].values.astype(float), tg["thrown"].values, tg["accepted"].values
    T, R, S, K, G = len(teams), len(refs), int(s_i.max() + 1), len(COVS), int(g_i.max() + 1)
    cores = cores or max(1, min(chains, os.cpu_count() or 1))
    with pm.Model():
        a0 = pm.Normal("a0", np.log(7.2), 0.3)
        sd_rw = pm.HalfNormal("sd_rw", 0.10)
        season_eff = pm.Deterministic("season_eff", pt.concatenate([pt.zeros(1), pt.cumsum(sd_rw * pm.Normal("z_s", 0, 1, shape=S - 1))]))
        sd_team = pm.HalfNormal("sd_team", 0.10)
        sd_opp = pm.HalfNormal("sd_opp", 0.05)
        opp_eff = pm.Deterministic("opp_eff", sd_opp * pm.Normal("z_o", 0, 1, shape=T))
        sd_trw = pm.HalfNormal("sd_trw", 0.04)
        z0 = pm.Normal("z_t", 0, 1, shape=T)
        zinc = pm.Normal("z_tinc", 0, 1, shape=(T, S - 1))
        ts_eff = pm.Deterministic("ts_eff", sd_team * z0[:, None] + pt.concatenate([pt.zeros((T, 1)), pt.cumsum(sd_trw * zinc, axis=1)], axis=1))
        team_eff = pm.Deterministic("team_eff", pt.zeros(T))
        sd_ref = pm.HalfNormal("sd_ref", 0.08)
        ref_eff = pm.Deterministic("ref_eff", sd_ref * pm.Normal("z_r", 0, 1, shape=R))
        beta = pm.Normal("beta", 0, 0.15, shape=K)
        sd_wk = pm.HalfNormal("sd_wk", 0.05)
        week_eff = pm.Deterministic("week_eff", pt.concatenate([pt.zeros(1), pt.cumsum(sd_wk * pm.Normal("z_w", 0, 1, shape=17))]))
        sd_game = pm.HalfNormal("sd_game", 0.15)
        u = sd_game * pm.Normal("z_g", 0, 1, shape=G)
        alpha = pm.Gamma("alpha", alpha=3, beta=0.1)
        log_mu = (a0 + season_eff[s_i] + team_eff[t_i] + opp_eff[o_i] + ts_eff[t_i, s_i] + ref_eff[r_i]
                  + pt.dot(X, beta) + u[g_i] + week_eff[w_i])
        pm.NegativeBinomial("y", mu=pt.exp(log_mu), alpha=alpha, observed=y)
        c0 = pm.Normal("c0", 1.7, 0.5)
        sd_at = pm.HalfNormal("sd_at", 0.15)
        acc_team = pm.Deterministic("acc_team", sd_at * pm.Normal("z_at", 0, 1, shape=T))
        sd_ar = pm.HalfNormal("sd_ar", 0.15)
        acc_ref = pm.Deterministic("acc_ref", sd_ar * pm.Normal("z_ar", 0, 1, shape=R))
        p = pm.math.invlogit(c0 + acc_team[t_i] + acc_ref[r_i])
        pm.Binomial("acc", n=y, p=p, observed=acc)
        idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores, blas_cores=1,
                          nuts_sampler="nutpie" if _fast_sampler_available() else "pymc",
                          random_seed=seed, progressbar=False, target_accept=0.9)
    keep = ["a0", "season_eff", "team_eff", "opp_eff", "ts_eff", "ref_eff", "beta", "sd_game", "alpha", "c0", "acc_team",
            "acc_ref", "sd_rw", "sd_team", "sd_opp", "sd_trw", "sd_ref", "sd_at", "sd_ar", "week_eff", "sd_wk"]
    post = {}
    for k in keep:
        v = idata.posterior[k].values
        post[k] = v.reshape((-1,) + v.shape[2:])
    return post, idata, teams, refs


def extend_posterior(post, S_needed, seed=0):
    """Future seasons beyond the training data: continue the season-level and team-effect random walks."""
    S = post["season_eff"].shape[1]
    if S_needed <= S:
        return post
    rng = np.random.default_rng(seed + 1)          # separate stream: never perturbs the simulation draws
    k, D, T = S_needed - S, post["a0"].shape[0], post["ts_eff"].shape[1]
    p = dict(post)
    p["season_eff"] = np.concatenate([post["season_eff"], post["season_eff"][:, -1:] + np.cumsum(post["sd_rw"][:, None] * rng.standard_normal((D, k)), 1)], 1)
    p["ts_eff"] = np.concatenate([post["ts_eff"], post["ts_eff"][:, :, -1:] + np.cumsum(post["sd_trw"][:, None, None] * rng.standard_normal((D, T, k)), 2)], 2)
    return p


def simulate(post, meta, g, nrep=12, seed=7, crew_shrink=0.6):
    """
    Posterior-predictive simulation. `g` = one row per game: home_team, away_team, season, week, referee (NaN = unassigned),
    dome, div_game, primetime, wind10, cold. Returns int arrays (n_sims, n_games): fh, fa (flags thrown), ah, aa (accepted).
    """
    rng = np.random.default_rng(seed)
    tix, rix = {t: i for i, t in enumerate(meta["teams"])}, {r: i for i, r in enumerate(meta["refs"])}
    fs = meta.get("first_season", FIRST_SEASON)
    post = extend_posterior(post, int(g.season.max()) - fs + 1, seed)
    D, G = post["a0"].shape[0], len(g)
    ht, at = g.home_team.map(tix).values, g.away_team.map(tix).values
    ss, ww = g.season.values - fs, (g.week.values - 1).astype(int)
    rr_s = g.referee.map(rix)
    unk = rr_s.isna().values                       # crew not (yet) assigned -> marginalise over the active crews
    rr = rr_s.fillna(0).astype(int).values
    base_cov = {c: g[c].values.astype(float) for c in COVS if c != "home"}
    b, ci = post["beta"], {c: i for i, c in enumerate(COVS)}
    cov_part = sum(b[:, ci[c]][:, None] * base_cov[c][None, :] for c in base_cov)
    home_b = b[:, ci["home"]][:, None]
    sidx = np.arange(D)[:, None]
    if unk.any():
        act = np.array([rix[a] for a in meta["active_refs"] if a in rix])
        rr_sim = np.where(unk[None, :], rng.choice(act, size=(D, G)), rr[None, :])      # a random active crew per simulation
    else:
        rr_sim = np.broadcast_to(rr[None, :], (D, G))
    common = (post["a0"][:, None] + post["season_eff"][sidx, ss[None, :]] + crew_shrink * post["ref_eff"][sidx, rr_sim] + cov_part)
    common = common + post["week_eff"][sidx, ww[None, :]]
    u = post["sd_game"][:, None] * rng.standard_normal((D, G))                         # game effect shared by both teams

    def side(team, opp, is_home):
        lm = (common + post["team_eff"][sidx, team[None, :]] + post["opp_eff"][sidx, opp[None, :]]
              + post["ts_eff"][sidx, team[None, :], ss[None, :]] + u + (home_b if is_home else 0.0))
        return np.exp(lm)

    mu_h, mu_a = side(ht, at, True), side(at, ht, False)
    alpha = post["alpha"][:, None, None]

    def nb(mu):
        mu = mu[:, None, :] * np.ones((1, nrep, 1))
        return rng.poisson(rng.gamma(shape=alpha, scale=mu / alpha))

    fh, fa = nb(mu_h), nb(mu_a)

    def acc_p(team):
        lg = post["c0"][:, None] + post["acc_team"][sidx, team[None, :]] + crew_shrink * post["acc_ref"][sidx, rr_sim]
        return (1 / (1 + np.exp(-lg)))[:, None, :] * np.ones((1, nrep, 1))

    ah, aa = rng.binomial(fh, acc_p(ht)), rng.binomial(fa, acc_p(at))
    flat = lambda a: a.reshape(D * nrep, G).astype(np.int16)
    return dict(fh=flat(fh), fa=flat(fa), ah=flat(ah), aa=flat(aa))


def _expected_for_crews(post, meta, g, rr, crew_shrink, seed):
    """Noise-free expected [flags thrown home, away, accepted home, away] per game for a given crew index per game (G,)."""
    tix = {t: i for i, t in enumerate(meta["teams"])}
    fs = meta.get("first_season", FIRST_SEASON)
    post = extend_posterior(post, int(g.season.max()) - fs + 1, seed)
    D, G = post["a0"].shape[0], len(g)
    ht, at = g.home_team.map(tix).values, g.away_team.map(tix).values
    ss, ww = g.season.values - fs, (g.week.values - 1).astype(int)
    b, ci = post["beta"], {c: i for i, c in enumerate(COVS)}
    cov = sum(b[:, ci[c]][:, None] * g[c].values.astype(float)[None, :] for c in COVS if c != "home")
    sidx = np.arange(D)[:, None]
    common = (post["a0"][:, None] + post["season_eff"][sidx, ss[None, :]] + crew_shrink * post["ref_eff"][:, rr] + cov
              + post["week_eff"][sidx, ww[None, :]])
    gm = np.exp(0.5 * post["sd_game"] ** 2)[:, None]                       # E[exp(game effect)]

    def mu(team, opp, home):
        lm = common + post["team_eff"][sidx, team[None, :]] + post["opp_eff"][sidx, opp[None, :]] + post["ts_eff"][sidx, team[None, :], ss[None, :]]
        return np.exp(lm + (b[:, ci["home"]][:, None] if home else 0.0)) * gm

    def p(team):
        return 1 / (1 + np.exp(-(post["c0"][:, None] + post["acc_team"][sidx, team[None, :]] + crew_shrink * post["acc_ref"][:, rr])))

    mh, ma = mu(ht, at, True), mu(at, ht, False)
    return np.stack([mh.mean(0), ma.mean(0), (mh * p(ht)).mean(0), (ma * p(at)).mean(0)])


def crew_deltas(post, meta, g, crew_shrink=0.6, seed=7):
    """Per game: crew effect on expected [game thrown, home thrown, away thrown, game accepted, home acc, away acc] relative to an
    unassigned crew (average over the active crews). None for games without an assigned crew. Exact: no simulation noise."""
    rix = {r: i for i, r in enumerate(meta["refs"])}
    rr_s = g.referee.map(rix)
    if not rr_s.notna().any():
        return [None] * len(g)
    act = [rix[a] for a in meta["active_refs"] if a in rix]
    avg = np.mean([_expected_for_crews(post, meta, g, np.full(len(g), j), crew_shrink, seed) for j in act], axis=0)
    ass = _expected_for_crews(post, meta, g, rr_s.fillna(0).astype(int).values, crew_shrink, seed)
    d = ass - avg                                                              # (4, G): Th, Ta, Ah, Aa
    out = []
    for i in range(len(g)):
        out.append(None if pd.isna(rr_s.iloc[i]) else [d[0, i] + d[1, i], d[0, i], d[1, i], d[2, i] + d[3, i], d[2, i], d[3, i]])
    return out


# =====================================================================================================================
# 3. PRICING: simulated distribution -> sportsbook-style line + two-way odds
# =====================================================================================================================
def american_from_prob(q):
    q = float(np.clip(q, 0.03, 0.97))
    return -100.0 * q / (1.0 - q) if q >= 0.5 else 100.0 * (1.0 - q) / q


def round_american(o, step=5):
    r = max(int(step * round(abs(o) / step)), 100)
    return -r if o < 0 else r


def prob_from_american(o):
    o = float(o)
    return -o / (-o + 100.0) if o < 0 else 100.0 / (o + 100.0)


def decimal_from_american(o):
    o = float(o)
    return 1.0 + (100.0 / -o if o < 0 else o / 100.0)


def quote(p_over, hold):
    """Multiplicative margin on fair probabilities: q_i = p_i (1 + hold); American odds in 5-cent steps."""
    oo = round_american(american_from_prob(p_over * (1 + hold)))
    uu = round_american(american_from_prob((1 - p_over) * (1 + hold)))
    return oo, uu, prob_from_american(oo) + prob_from_american(uu) - 1.0


def price_market(sims, hold=0.07, n_alts=2, min_p=0.04):
    """Main line = half-point closest to 50/50 (skew goes into the odds); optional alternate lines either side."""
    s = np.asarray(sims).astype(int)
    lo = s.min()
    pmf = np.bincount(s - lo) / s.size
    p_over_all = 1.0 - np.cumsum(pmf)                      # P(s > x + 0.5)
    lines = lo + np.arange(len(pmf)) + 0.5
    ok = np.where((p_over_all > min_p) & (p_over_all < 1 - min_p))[0]
    k_main = int(ok[np.argmin(np.abs(p_over_all[ok] - 0.5))]) if len(ok) else int(np.argmin(np.abs(p_over_all - 0.5)))
    out = []
    for off in [0] + [d for n in range(1, n_alts + 1) for d in (-n, n)]:
        k = k_main + off
        if 0 <= k < len(lines) and min_p < p_over_all[k] < 1 - min_p:
            p = float(p_over_all[k])
            oo, uu, hr = quote(p, hold)
            out.append(dict(kind="main" if off == 0 else "alt", line=float(lines[k]), over_odds=oo, under_odds=uu,
                            p_over=p, p_under=1 - p, hold_realised=hr))
    return out


def summarize(s):
    s = np.asarray(s)
    q = np.percentile(s, [5, 10, 25, 50, 75, 90, 95])
    return dict(mean=float(s.mean()), sd=float(s.std()), median=float(q[3]), p05=float(q[0]), p10=float(q[1]), p25=float(q[2]),
                p75=float(q[4]), p90=float(q[5]), p95=float(q[6]))


# =====================================================================================================================
# 4. MODEL FILES, INPUT HANDLING
# =====================================================================================================================
def crew_table(post, meta, shrink=0.6):
    """Estimated referee-crew effects on the FLAG RATE (log-scale posterior -> %). 'vs_active_avg' is what moves a line when a crew is assigned
    instead of an unknown crew (which the model averages over the active crews)."""
    refs, re = meta["refs"], post["ref_eff"]
    act = [refs.index(r) for r in meta["active_refs"] if r in refs]
    base = np.exp(shrink * re[:, act]).mean(1)              # per-draw average of the active crews' multipliers
    rows = []
    for j, r in enumerate(refs):
        if r.startswith("UNK_"):
            continue
        x = re[:, j]
        rows.append(dict(referee=r, games=meta.get("ref_games", {}).get(r), last_season=meta.get("ref_last_season", {}).get(r),
                         active_2026=r in meta["active_refs"], raw_pct=100 * (np.exp(x).mean() - 1),
                         raw_lo80=100 * (np.exp(np.percentile(x, 10)) - 1), raw_hi80=100 * (np.exp(np.percentile(x, 90)) - 1),
                         applied_pct=100 * (np.exp(shrink * x).mean() - 1),
                         vs_active_avg_pct=100 * ((np.exp(shrink * x) / base).mean() - 1)))
    return pd.DataFrame(rows).sort_values("vs_active_avg_pct", ascending=False).reset_index(drop=True)


def load_model(model_dir=MODEL_DIR):
    mp, pp = Path(model_dir) / "meta.json", Path(model_dir) / "posterior.npz"
    if not mp.exists() or not pp.exists():
        raise SystemExit(f"no trained model in {model_dir}. Run: python3 nfl_flags.py update")
    meta = json.load(open(mp))
    if meta["covs"] != COVS:
        raise SystemExit("model/meta.json was built with different covariates than this script expects")
    return meta, dict(np.load(pp))


def norm_team(t, meta):
    t = TEAM_MAP.get(str(t).upper(), str(t).upper())
    if t not in meta["teams"]:
        raise SystemExit(f"unknown team '{t}'. Valid: {' '.join(meta['teams'])}")
    return t


def resolve_ref(name, meta):
    if name is None or (isinstance(name, float) and np.isnan(name)) or str(name).strip() == "":
        return np.nan
    real = [r for r in meta["refs"] if not r.startswith("UNK_")]
    low = {r.lower(): r for r in real}
    if str(name).lower() in low:
        return low[str(name).lower()]
    m = difflib.get_close_matches(str(name), real, n=1, cutoff=0.75)
    if m:
        print(f"note: referee '{name}' matched to '{m[0]}'")
        return m[0]
    print(f"note: referee '{name}' not in the model's history; treating the crew as unassigned")
    return np.nan


def games_frame(rows, meta):
    """rows: dicts with home, away, season, week (+ optional referee, dome, div_game, primetime, temp, wind, roof, gameday, gametime, game_id)."""
    out = []
    for r in rows:
        h, a = norm_team(r["home"], meta), norm_team(r["away"], meta)
        if h == a:
            raise SystemExit("home and away teams must differ")
        if not 1 <= int(r["week"]) <= 18:
            raise SystemExit("week must be 1-18")
        roof = {"dome": 1, "closed": 1, "outdoors": 0, "open": 0}.get(r.get("roof"))
        dome = r.get("dome")
        if dome is None or (isinstance(dome, float) and np.isnan(dome)):
            dome = roof if roof is not None else meta.get("team_dome", {}).get(h, int(h in FIXED_ROOF))
        dome = int(dome)
        temp = 60.0 if r.get("temp") is None or pd.isna(r.get("temp")) else float(r["temp"])
        wind = meta["typical_wind"] if r.get("wind") is None or pd.isna(r.get("wind")) else float(r["wind"])
        if dome:
            temp, wind = 70.0, 0.0
        div = r.get("div_game")
        div = int(TEAM_DIV[h] == TEAM_DIV[a]) if div is None or pd.isna(div) else int(div)
        pt_ = r.get("primetime")
        if pt_ is None or pd.isna(pt_):
            hr = pd.to_datetime(r.get("gametime"), format="%H:%M", errors="coerce")
            pt_ = int(hr.hour >= 19) if pd.notna(hr) else 0
        out.append(dict(game_id=r.get("game_id") or f"{int(r['season'])}_{int(r['week']):02d}_{a}_{h}", home_team=h, away_team=a,
                        season=int(r["season"]), week=int(r["week"]), referee=resolve_ref(r.get("referee"), meta), dome=dome, div_game=div,
                        primetime=int(pt_), temp=temp, wind=wind, wind10=wind / 10.0, cold=int(temp < 40 and not dome),
                        gameday=r.get("gameday", ""), gametime=r.get("gametime", ""), played=bool(r.get("played", False))))
    return pd.DataFrame(out)


def stale_warning(meta, sched):
    if sched is None:
        return
    done = sched[(sched.game_type == "REG") & sched.home_score.notna()]
    if done.empty:
        return
    latest = (int(done.season.max()), int(done[done.season == done.season.max()].week.max()))
    if latest > (meta["last_season"], meta["last_week"]):
        n = int(((done.season > meta["last_season"]) | ((done.season == meta["last_season"]) & (done.week > meta["last_week"]))).sum())
        print(f"note: model trained through {meta['last_season']} wk{meta['last_week']}; {n} newer games have been played "
              f"(latest {latest[0]} wk{latest[1]}). Run `python3 nfl_flags.py update` to refit.\n")


def market_series(sims, i, h, a):
    """(key, label, team, array) for the six main markets of game i."""
    return [("game_thrown", "Flags thrown (game)", "", sims["fh"][:, i] + sims["fa"][:, i]),
            ("team_thrown", f"{h} flags thrown", h, sims["fh"][:, i]), ("team_thrown", f"{a} flags thrown", a, sims["fa"][:, i]),
            ("game_accepted", "Accepted (game)", "", sims["ah"][:, i] + sims["aa"][:, i]),
            ("team_accepted", f"{h} accepted", h, sims["ah"][:, i]), ("team_accepted", f"{a} accepted", a, sims["aa"][:, i])]


def line_rows(g, sims, hold, alts, deltas=None):
    rows = []
    for i, r in g.iterrows():
        crew = r.referee if isinstance(r.referee, str) else "TBD (active-crew mix)"
        kick = f"{r.gameday} {r.gametime}".strip()
        for j, (mk, label, team, s) in enumerate(market_series(sims, i, r.home_team, r.away_team)):
            delta = round(float(deltas[i][j]), 2) if deltas is not None and deltas[i] is not None else None
            for q in price_market(s, hold, alts):
                rows.append(dict(game_id=r.game_id, crew_effect_flags=delta, kickoff_et=kick, matchup=f"{r.away_team} @ {r.home_team}", referee=crew, market=mk,
                                 team=team, kind=q["kind"], line=q["line"], over_odds=q["over_odds"], under_odds=q["under_odds"],
                                 model_mean=round(float(s.mean()), 2), model_median=float(np.median(s)), p_over_fair=round(q["p_over"], 4),
                                 hold_realised=round(q["hold_realised"], 4), book="MODEL"))
    return pd.DataFrame(rows)


# =====================================================================================================================
# 5. COMMANDS
# =====================================================================================================================
def report(g, sims, i, hold, alts, args=None, deltas=None, crew_row=None):
    r = g.iloc[i]
    crew = r.referee if isinstance(r.referee, str) else "TBD (active-crew mix)"
    when = f" | {r.gameday} {r.gametime} ET".rstrip() if r.gameday else ""
    print(f"{r.away_team} @ {r.home_team} | {r.season} week {r.week}{when} | crew: {crew}")
    print(f"context: dome={r.dome} divisional={r.div_game} primetime={r.primetime} temp={r.temp:g}F wind={r.wind:g}mph")
    if deltas is not None and deltas[i] is not None:
        d_t, d_a = deltas[i][0], deltas[i][3]
        extra = ""
        if crew_row is not None:
            extra = (f"; crew rate effect raw {crew_row.raw_pct:+.1f}% [80%: {crew_row.raw_lo80:+.1f} to {crew_row.raw_hi80:+.1f}], "
                     f"{'' if pd.isna(crew_row.games) else f'{int(crew_row.games)} games of history, '}applied x{args.crew_shrink if args is not None else ''}")
        print(f"CREW EFFECT vs an average active crew: flags thrown {d_t:+.2f}, accepted {d_a:+.2f}{extra}")
    if r.played:
        print("WARNING: this game has been played and the model was trained on it, so this is IN-SAMPLE (not a forecast).")
    print()
    for title, keys in [("FLAGS THROWN", ("game_thrown", "team_thrown")), ("PENALTIES ACCEPTED", ("game_accepted", "team_accepted"))]:
        items = [x for x in market_series(sims, i, r.home_team, r.away_team) if x[0] in keys]
        tot = summarize(items[0][3])
        print(f"{title}: expected {tot['mean']:.1f}  (median {tot['median']:g}; 50% {tot['p25']:g}-{tot['p75']:g}; "
              f"80% {tot['p10']:g}-{tot['p90']:g}; 90% {tot['p05']:g}-{tot['p95']:g})")
        for _, lab, team, s in items[1:]:
            z = summarize(s)
            print(f"    {team:<4} {z['mean']:.1f}  (80%: {z['p10']:g}-{z['p90']:g})")
    print(f"\nPROP LINES  (hold {hold:.0%}; line, over/under American odds)")
    for k, lab, team, s in market_series(sims, i, r.home_team, r.away_team):
        qs = price_market(s, hold, alts)
        main = qs[0]
        print(f"  {lab:<26} {main['line']:>5g}  {main['over_odds']:+d}/{main['under_odds']:+d}   (fair P(over) {main['p_over']:.1%}, mean {s.mean():.2f})")
        for q in sorted(qs[1:], key=lambda q: q["line"]):
            print(f"      alt {q['line']:>5g}  {q['over_odds']:+d}/{q['under_odds']:+d}   (fair P(over) {q['p_over']:.1%})")
    if args is not None and args.line is not None:
        key = {"game_thrown": 0, "game_accepted": 3, "home_thrown": 1, "away_thrown": 2, "home_accepted": 4, "away_accepted": 5}[args.market]
        s = market_series(sims, i, r.home_team, r.away_team)[key][3]
        L = args.line
        po, pp, pu = float((s > L).mean()), float((s == L).mean()), float((s < L).mean())
        print(f"\nYOUR LINE: {market_series(sims, i, r.home_team, r.away_team)[key][1]} {L:g}:  P(over) {po:.1%}  P(push) {pp:.1%}  P(under) {pu:.1%}")
        fair = lambda p: f"{american_from_prob(p / max(1 - pp, 1e-9)):+.0f}"
        print(f"   fair odds (push excluded): over {fair(po)} / under {fair(pu)}")
        if args.odds:
            oo, uu = [float(x) for x in args.odds.split("/")]
            ev_o = po * (decimal_from_american(oo) - 1) - pu
            ev_u = pu * (decimal_from_american(uu) - 1) - po
            print(f"   at {oo:+.0f}/{uu:+.0f}: model EV per $1 -> over {ev_o:+.3f}, under {ev_u:+.3f}  "
                  f"(model probabilities are overconfident in backtests; discount heavily)")
    print(f"\n{DISCLAIMER}")


def cmd_predict(a):
    meta, post = load_model(a.model_dir)
    sched = None
    if a.game_id:
        sched = load_schedule(a.data_dir, refresh=not a.offline)
        if sched is None or a.game_id not in set(sched.game_id):
            raise SystemExit(f"game_id {a.game_id} not found in schedule")
        s = sched[sched.game_id == a.game_id].iloc[0].to_dict()
        row = dict(home=s["home_team"], away=s["away_team"], season=s["season"], week=s["week"], referee=a.referee or s.get("referee"),
                   roof=s.get("roof"), temp=a.temp if a.temp is not None else s.get("temp"), wind=a.wind if a.wind is not None else s.get("wind"),
                   div_game=s.get("div_game"), gameday=s.get("gameday", ""), gametime=s.get("gametime", ""), game_id=a.game_id,
                   played=pd.notna(s.get("home_score")))
    else:
        if not (a.home and a.away and a.week):
            raise SystemExit("give --game-id, or --home, --away and --week")
        row = dict(home=a.home, away=a.away, season=a.season or meta["last_season"], week=a.week, referee=a.referee, temp=a.temp, wind=a.wind,
                   dome=(1 if a.dome else 0 if a.outdoors else None), primetime=(1 if a.primetime else None))
        if not a.offline:
            sched = load_schedule(a.data_dir, refresh=True, quiet=True)
    stale_warning(meta, sched)
    g = games_frame([row], meta)
    sims = simulate(post, meta, g, nrep=a.nrep, seed=a.seed, crew_shrink=a.crew_shrink)
    deltas, crew_row = crew_deltas(post, meta, g, a.crew_shrink, a.seed), None
    if isinstance(g.referee.iloc[0], str):
        ct = crew_table(post, meta, a.crew_shrink)
        crew_row = ct[ct.referee == g.referee.iloc[0]].iloc[0]
    if a.json:
        out = {"game": g.iloc[0].drop("played").to_dict(), "markets": {}}
        for k, lab, team, s in market_series(sims, 0, g.home_team[0], g.away_team[0]):
            out["markets"][lab] = {**summarize(s), "lines": price_market(s, a.hold, a.alts)}
        if deltas[0] is not None:
            out["crew_effect_flags_thrown"] = float(deltas[0][0])
        print(json.dumps(out, indent=2, default=lambda o: o if not isinstance(o, np.generic) else o.item()))
        return
    report(g, sims, 0, a.hold, a.alts, a, deltas, crew_row)


def write_week_md(path, g, L, meta, post, a):
    """Readable lines sheet: one row per game, plus the crew-effect table for the crews actually assigned."""
    main = L[L.kind == "main"]
    f = lambda x: f"{x.line:g} ({x.over_odds:+d}/{x.under_odds:+d})"
    out = [f"# {a.season} Week {a.week} penalty prop lines (model-generated; not sportsbook data)", "",
           f"Model trained through {meta['last_season']} week {meta['last_week']} ({meta['n_rows']:,} team-games, 1999+). Hold {a.hold:.0%}, crew effects scaled to {a.crew_shrink:.0%}. "
           "Matchups read AWAY @ HOME; line (over/under American odds); H = home team, A = away team.", "",
           "| Matchup | Kickoff (ET) | Crew | Crew effect (flags) | Thrown: game | Thrown: H | Thrown: A | Accepted: game | Accepted: H | Accepted: A |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in g.iterrows():
        m = main[main.game_id == r.game_id]
        pick = lambda mk, t="": f(m[(m.market == mk) & (m.team == t)].iloc[0])
        ce = m[m.market == "game_thrown"].crew_effect_flags.iloc[0]
        out.append(f"| {r.away_team} @ {r.home_team} | {str(r.gameday)[5:]} {r.gametime} | {m.referee.iloc[0]} | "
                   f"{'' if pd.isna(ce) else f'{ce:+.1f}'} | {pick('game_thrown')} | {pick('team_thrown', r.home_team)} | {pick('team_thrown', r.away_team)} | "
                   f"{pick('game_accepted')} | {pick('team_accepted', r.home_team)} | {pick('team_accepted', r.away_team)} |")
    assigned = [(r.game_id, r.referee, f"{r.away_team} @ {r.home_team}") for _, r in g.iterrows() if isinstance(r.referee, str)]
    if assigned:
        ct = crew_table(post, meta, a.crew_shrink).set_index("referee")
        out += ["", "## Crew effects for the assigned crews", "",
                "Raw effect = posterior mean change in a game's flag rate attributable to the crew (80% interval). 'vs avg crew' is what the assignment changes relative to "
                f"an unknown crew after the {a.crew_shrink:.0%} scaling; 'this game' is the resulting change in expected flags thrown for that game.", "",
                "| Crew | Game | Games of history | Raw effect [80%] | vs avg crew | This game (flags thrown) |", "|---|---|---|---|---|---|"]
        for gid, ref, mu in assigned:
            c = ct.loc[ref]
            ce = main[(main.game_id == gid) & (main.market == "game_thrown")].crew_effect_flags.iloc[0]
            gp = "" if pd.isna(c.games) else int(c.games)
            out.append(f"| {ref} | {mu} | {gp} | {c.raw_pct:+.1f}% [{c.raw_lo80:+.1f}, {c.raw_hi80:+.1f}] | {c.vs_active_avg_pct:+.1f}% | {ce:+.2f} |")
    out += ["", "## Notes", "",
            "* Crew = head referee; the NFL may substitute members of a crew week to week. Crews with little history (e.g. second-season referees) have wide intervals.",
            "* Weather is neutral (60F, typical wind) unless supplied. Team markets are lumpy, so their skew is carried in the odds. Alternate lines are in the CSV.",
            f"* {DISCLAIMER}", ""]
    Path(path).write_text("\n".join(out))


def cmd_week(a):
    meta, post = load_model(a.model_dir)
    sched = load_schedule(a.data_dir, refresh=not a.offline)
    if sched is None:
        raise SystemExit("schedule unavailable (offline and no cached data/games.csv)")
    stale_warning(meta, sched)
    s = sched[(sched.season == a.season) & (sched.week == a.week) & (sched.game_type == "REG")]
    if not a.include_played:
        s = s[s.home_score.isna()]
    if s.empty:
        raise SystemExit("no (unplayed) games found for that season/week")
    rows = [dict(home=r.home_team, away=r.away_team, season=r.season, week=r.week, referee=r.referee, roof=r.roof, temp=r.temp, wind=r.wind,
                 div_game=r.div_game, gameday=r.gameday, gametime=r.gametime, game_id=r.game_id, played=pd.notna(r.home_score))
            for r in s.itertuples()]
    assign = {}
    if a.referees_file:
        rf = pd.read_csv(a.referees_file)
        assign.update({r.game_id: r.referee for r in rf.itertuples() if pd.notna(r.referee)})
    for kv in a.referee:
        gid, name = kv.split("=", 1)
        assign[gid] = name.strip('"')
    unknown_ids = set(assign) - {r["game_id"] for r in rows}
    if unknown_ids:
        print(f"note: referee assignments for games not in this slate were ignored: {sorted(unknown_ids)}")
    for r in rows:
        if r["game_id"] in assign:
            r["referee"] = assign[r["game_id"]]
    g = games_frame(rows, meta)
    sims = simulate(post, meta, g, nrep=a.nrep, seed=a.seed, crew_shrink=a.crew_shrink)
    deltas = crew_deltas(post, meta, g, a.crew_shrink, a.seed)
    L = line_rows(g, sims, a.hold, a.alts, deltas)
    L.to_csv(a.out, index=False)
    main = L[L.kind == "main"]
    print(f"{len(g)} games | {sims['fh'].shape[0]} sims each | hold {a.hold:.0%} | crew shrink {a.crew_shrink} | wrote {a.out} ({len(L)} lines)\n")
    f = lambda x: f"{x.line:g} ({x.over_odds:+d}/{x.under_odds:+d})"
    for _, r in g.iterrows():
        m = main[main.game_id == r.game_id]
        pick = lambda mk, t="": f(m[(m.market == mk) & (m.team == t)].iloc[0])
        ce = m[m.market == "game_thrown"].crew_effect_flags.iloc[0]
        ce_txt = f"  ({ce:+.1f} flags vs avg crew)" if pd.notna(ce) else ""
        print(f"{r.away_team:>3} @ {r.home_team:<3} {str(r.gameday)[5:]} {str(r.gametime):>5}  crew: {m.referee.iloc[0]}{ce_txt}")
        print(f"     thrown   total {pick('game_thrown'):<18} {r.home_team} {pick('team_thrown', r.home_team):<18} {r.away_team} {pick('team_thrown', r.away_team)}")
        print(f"     accepted total {pick('game_accepted'):<18} {r.home_team} {pick('team_accepted', r.home_team):<18} {r.away_team} {pick('team_accepted', r.away_team)}")
    if a.md:
        write_week_md(a.md, g, L, meta, post, a)
        print(f"\nwrote readable sheet: {a.md}")
    print(f"\n{DISCLAIMER}")


def _fast_sampler_available():
    """nutpie compiles PyMC models with numba, so it is only usable when BOTH are installed."""
    return importlib.util.find_spec("nutpie") is not None and importlib.util.find_spec("numba") is not None


def check_update_dependencies():
    """Fail fast (before any download) if the packages needed to FIT the model are missing."""
    need = [p for p in ("pymc", "arviz", "pyarrow") if importlib.util.find_spec(p) is None]
    if need:
        py = sys.executable
        raise SystemExit(
            f"\n`update` needs these Python packages, which are not installed: {', '.join(need)}\n\n"
            "Pick ONE route:\n"
            "  A) Recommended (fastest): a Python 3.11-3.13 environment (3.12 is best; NOT 3.14)\n"
            "       python3.12 -m venv .venv && source .venv/bin/activate && pip install -r requirements-refit.txt\n"
            "  B) Stay on your current Python (including 3.14) and avoid numba/llvmlite (slower):\n"
            f"       {py} -m pip install -r requirements-refit-nonumba.txt\n"
            "       then:  python3 nfl_flags.py update --since 2018 --chains 2\n\n"
            "If pip failed with 'Failed building wheel for llvmlite': the newest llvmlite/numba have no prebuilt wheels for Intel Macs (and none at all for\n"
            "Python 3.14), so pip tries to compile LLVM. requirements-refit.txt pins the last versions that DO have wheels (numba<0.63, llvmlite<0.46) and works\n"
            "on Python 3.11-3.13. Removing nutpie alone does NOT avoid it, because ArviZ 1.x / PyTensor also need numba. Use route A, or route B.\n"
            "Data already downloaded is kept, so rerunning `update` will not download it again.\n"
            "`predict`, `week` and `crews` do not need any of this.")
    if not _fast_sampler_available():
        print("[fit] note: nutpie and/or numba is not installed; using PyMC's default sampler (works, but slower; it needs a C compiler, on macOS `xcode-select --install`).\n"
              "      Tip: `update --since 2018 --chains 2` keeps a refit practical. nutpie needs numba/llvmlite, which do not install on every Python.")


def game_actuals(gid, season, data_dir, refresh=False, offline=False):
    """Flags actually called in one PLAYED game, parsed from the play-by-play text (same method and de-duplication as the training data)."""
    path = Path(data_dir) / f"play_by_play_{season}.parquet"
    if refresh or not path.exists():
        if offline:
            raise SystemExit(f"{path} is missing and --offline was given. Run without --offline, or `python3 nfl_flags.py update`, to download it.")
        print(f"[data] downloading {season} play-by-play ...")
        _download(PBP_URL.format(y=season), path)
    try:
        d = pd.read_parquet(path, columns=["game_id", "play_id", "qtr", "time", "desc", "penalty"], filters=[("game_id", "==", gid)])
    except ImportError:
        raise SystemExit("reading play-by-play needs pyarrow:  python3 -m pip install pyarrow")
    if d.empty:
        raise SystemExit(f"{gid} is not in {path.name}. If the game finished recently, rerun with --refresh to download the latest data. "
                         "(nflverse has no play-by-play at all for a handful of 1999-2000 games, e.g. 1999_01_BAL_STL.)")
    rows = []
    for pid, qtr, tm, desc in zip(d.play_id, d.qtr, d.time, d.desc.fillna("")):
        for m in PEN_RE.finditer(desc):
            rows.append(dict(play_id=pid, qtr=qtr, time=tm, team=TEAM_MAP.get(m.group("team"), m.group("team")),
                             type=m.group("ptype").strip(), status=_status(m.group("status"))))
    fl = pd.DataFrame(rows, columns=["play_id", "qtr", "time", "team", "type", "status"])
    fl = fl.drop_duplicates(["play_id", "team", "status"]).sort_values("play_id")
    return fl, int(d.penalty.sum())


def actual_totals(fl, home, away):
    out = {}
    for t in (home, away):
        x = fl[fl.team == t]
        out[t] = dict(thrown=len(x), accepted=int((x.status == "enforced").sum()), declined=int((x.status == "declined").sum()),
                      offsetting=int((x.status == "offsetting").sum()))
    out["game"] = {k: out[home][k] + out[away][k] for k in out[home]}
    return out


def _settle(lines_df, gid, T):
    """Yield (label, market, kind, line, over_odds, under_odds, actual, result) for each line of one game."""
    for x in lines_df[lines_df.game_id == gid].itertuples():
        if x.market.startswith("team"):
            act = T[x.team]["thrown" if x.market == "team_thrown" else "accepted"]
            lab = f"{x.team} flags thrown" if x.market == "team_thrown" else f"{x.team} accepted"
        else:
            act = T["game"]["thrown" if x.market == "game_thrown" else "accepted"]
            lab = "Flags thrown (game)" if x.market == "game_thrown" else "Accepted (game)"
        res = "OVER" if act > float(x.line) else "UNDER" if act < float(x.line) else "PUSH"
        yield lab, x.market, getattr(x, "kind", "main"), float(x.line), int(x.over_odds), int(x.under_odds), act, res


def actual_slate(a, sched):
    season = a.season or int(sched[sched.home_score.notna()].season.max())
    s = sched[(sched.season == season) & (sched.week == a.week) & (sched.game_type == "REG")]
    done, todo = s[s.home_score.notna()], s[s.home_score.isna()]
    if done.empty:
        raise SystemExit(f"no played games found for {season} week {a.week} yet. If games finished recently, rerun without --offline.")
    L = None
    if a.lines:
        L = pd.read_csv(a.lines, keep_default_na=False)
        if "kind" in L and not a.alts:
            L = L[L.kind == "main"]
    crew_of = {}
    if L is not None and "referee" in L:
        crew_of = {g: r for g, r in zip(L.game_id, L.referee) if isinstance(r, str) and not r.startswith("TBD")}
    print(f"{season} week {a.week}: {len(done)} played, {len(todo)} not yet played\n")
    print(f"{'game':<12}{'final':>10}   {'thrown':>7} {'(H/A)':>7}   {'accepted':>9} {'(H/A)':>7}   crew")
    rows = []
    for r in done.itertuples():
        fl, _ = game_actuals(r.game_id, int(r.season), a.data_dir, a.refresh, a.offline)
        T = actual_totals(fl, r.home_team, r.away_team)
        print(f"{r.away_team + ' @ ' + r.home_team:<12}{int(r.away_score):>4}-{int(r.home_score):<5}   {T['game']['thrown']:>7} {str(T[r.home_team]['thrown']) + '/' + str(T[r.away_team]['thrown']):>7}   "
              f"{T['game']['accepted']:>9} {str(T[r.home_team]['accepted']) + '/' + str(T[r.away_team]['accepted']):>7}   {r.referee if isinstance(r.referee, str) else crew_of.get(r.game_id, 'n/a')}")
        if L is not None:
            rows += [dict(game=r.game_id, label=lab, market=mk, line=ln, actual=act, result=res, over=ov, under=un) for lab, mk, kd, ln, ov, un, act, res in _settle(L, r.game_id, T)]
        a_tot = T["game"]
    print("\n'thrown' = accepted + declined + offsetting; picked-up flags are not in the data. 'accepted' matches how penalties are normally counted in box scores.")
    if L is not None and rows:
        R = pd.DataFrame(rows)
        print(f"\nSETTLED AGAINST {a.lines} ({R.game.nunique()} games):")
        print(f"{'market':<16}{'lines':>6}{'overs':>7}{'unders':>8}{'pushes':>8}{'avg line':>10}{'avg actual':>12}{'actual-line':>13}")
        for mk in ["game_thrown", "game_accepted", "team_thrown", "team_accepted"]:
            g = R[R.market == mk]
            if g.empty:
                continue
            print(f"{mk:<16}{len(g):>6}{(g.result == 'OVER').sum():>7}{(g.result == 'UNDER').sum():>8}{(g.result == 'PUSH').sum():>8}"
                  f"{g.line.mean():>10.2f}{g.actual.mean():>12.2f}{(g.actual - g.line).mean():>+13.2f}")
        n_g = R.game.nunique()
        print(f"\nA handful of games is far too few to judge a model: each game's total has a standard deviation of about 4.8 flags, so a {n_g}-game average has a standard error of about {4.8 / n_g ** 0.5:.1f} flags.")
        if a.show_lines:
            for gid, g in R.groupby("game", sort=False):
                print(f"\n{gid}")
                for x in g.itertuples():
                    print(f"  {x.label:<26} line {x.line:>5g} ({x.over:+d}/{x.under:+d})  actual {x.actual:>3}  -> {x.result}")
    if len(todo):
        print("\nnot yet played: " + ", ".join(f"{t.away_team}@{t.home_team}" for t in todo.itertuples()))


def cmd_actual(a):
    sched = load_schedule(a.data_dir, refresh=not a.offline)
    if sched is None:
        raise SystemExit("schedule unavailable (offline and no cached data/games.csv)")
    if not a.game_id and not (a.home and a.away):
        if not a.week:
            raise SystemExit("give --game-id, or --home --away --week, or just --week N to check every played game of that week")
        return actual_slate(a, sched)
    if a.game_id:
        row = sched[sched.game_id == a.game_id]
    else:
        if not (a.home and a.away and a.week):
            raise SystemExit("give --game-id, or --home, --away and --week")
        season = a.season or int(sched[sched.home_score.notna()].season.max())
        h, v = TEAM_MAP.get(a.home.upper(), a.home.upper()), TEAM_MAP.get(a.away.upper(), a.away.upper())
        row = sched[(sched.season == season) & (sched.week == a.week) & (sched.home_team == h) & (sched.away_team == v) & (sched.game_type == "REG")]
    if row.empty:
        raise SystemExit("game not found in the schedule (check the id; home team goes last, e.g. 2026_06_SEA_DEN)")
    r = row.iloc[0]
    if pd.isna(r.home_score):
        raise SystemExit(f"{r.game_id} has no final score yet, so there is nothing to check. If it finished recently, rerun without --offline to refresh the schedule.")
    home, away, gid = r.home_team, r.away_team, r.game_id
    fl, fast_acc = game_actuals(gid, int(r.season), a.data_dir, a.refresh, a.offline)
    T = actual_totals(fl, home, away)
    crew = r.referee if isinstance(r.referee, str) else "n/a (not in the schedule file yet)"
    print(f"{away} @ {home} | {int(r.season)} week {int(r.week)} | final {away} {int(r.away_score)} - {home} {int(r.home_score)} | crew: {crew}\n")
    print(f"{'':<12}{'thrown':>8}{'accepted':>10}{'declined':>10}{'offsetting':>12}")
    for lab, k in [(home, home), (away, away), ("GAME TOTAL", "game")]:
        v = T[k]
        print(f"{lab:<12}{v['thrown']:>8}{v['accepted']:>10}{v['declined']:>10}{v['offsetting']:>12}")
    print(f"\n'thrown' = accepted + declined + offsetting (each offsetting flag counts once per team). Picked-up flags are not in the data, so they are not counted.\n"
          f"cross-check: nflfastR's own penalty column shows {fast_acc} accepted for this game; this tool counts {T['game']['accepted']} "
          "(a difference of 1-2 usually means a play with two enforced penalties, which nflfastR records once).")
    if a.list:
        print("\nEvery flag found:")
        for x in fl.itertuples():
            print(f"  Q{int(x.qtr) if pd.notna(x.qtr) else '?'} {x.time!s:>5}  {x.team:<4} {x.type:<34} {x.status}")
    if a.lines:
        L = pd.read_csv(a.lines, keep_default_na=False)
        L = L[L.game_id == gid]
        if "kind" in L and not a.alts:
            L = L[L.kind == "main"]
        if L.empty:
            print(f"\n(no lines for {gid} in {a.lines})")
        else:
            print(f"\nSETTLED AGAINST {a.lines}:")
            for x in L.itertuples():
                if x.market.startswith("team"):
                    act = T[x.team]["thrown" if x.market == "team_thrown" else "accepted"]
                    lab = f"{x.team} {'flags thrown' if x.market == 'team_thrown' else 'accepted'}"
                else:
                    act = T["game"]["thrown" if x.market == "game_thrown" else "accepted"]
                    lab = "Flags thrown (game)" if x.market == "game_thrown" else "Accepted (game)"
                res = "OVER" if act > float(x.line) else "UNDER" if act < float(x.line) else "PUSH"
                tag = "" if getattr(x, "kind", "main") == "main" else " (alt)"
                print(f"  {lab + tag:<28} line {float(x.line):>5g} ({int(x.over_odds):+d}/{int(x.under_odds):+d})   actual {act:>3}   -> {res}")


def cmd_crews(a):
    meta, post = load_model(a.model_dir)
    ct = crew_table(post, meta, a.crew_shrink)
    if not a.all:
        ct = ct[ct.active_2026]
    ct = ct.copy()
    ct["games"] = ct.games.map(lambda x: "" if pd.isna(x) else int(x))
    ct["last_season"] = ct.last_season.map(lambda x: "" if pd.isna(x) else int(x))
    print(f"Referee-crew effects on the flag rate (model trained through {meta['last_season']} wk{meta['last_week']}; crew shrink {a.crew_shrink}).")
    print("raw = posterior mean effect with 80% interval; vs_active_avg = what an assignment changes relative to an unknown crew (applied shrink).\n")
    f = ct.assign(raw=lambda d: d.apply(lambda r: f"{r.raw_pct:+.1f}% [{r.raw_lo80:+.1f}, {r.raw_hi80:+.1f}]", axis=1),
                  vs_active_avg=lambda d: d.vs_active_avg_pct.map(lambda x: f"{x:+.1f}%"),
                  approx_flags=lambda d: d.vs_active_avg_pct.map(lambda x: f"{x * 0.165:+.1f}"))
    print(f[["referee", "games", "last_season", "raw", "vs_active_avg", "approx_flags"]].rename(columns={"approx_flags": "~flags/game"}).to_string(index=False))
    print("\nA crew with few games of history is shrunk toward zero by the model, but its estimate is still noisy (see the 80% interval).")
    if a.csv:
        ct.to_csv(a.csv, index=False)
        print(f"wrote {a.csv}")


def cmd_update(a):
    """Download latest data, rebuild the team-game table, refit, save model/posterior.npz + meta.json."""
    check_update_dependencies()
    data = Path(a.data_dir)
    data.mkdir(parents=True, exist_ok=True)
    sched = load_schedule(data, refresh=True)
    if sched is None:
        raise SystemExit("\nCould not get the NFL schedule (games.csv), so `update` cannot continue. See the message above. "
                         "If it is a certificate error, apply the fix and rerun; otherwise check your internet connection.")
    cur = int(sched[sched.home_score.notna()].season.max())
    for y in range(max(FIRST_SEASON, a.since), cur + 1):
        p = data / f"play_by_play_{y}.parquet"
        if y >= cur or a.refresh_all or not p.exists():
            print(f"[data] downloading {y} ...")
            _download(PBP_URL.format(y=y), p)
    tg, games = build_team_games(data, a.since)
    tg.to_parquet(data / "team_games.parquet")
    last = tg[tg.season == tg.season.max()]
    print(f"[data] {len(tg):,} team-games, {tg.game_id.nunique():,} games, through {int(tg.season.max())} wk{int(last.week.max())}")
    t0 = time.time()
    post, idata, teams, refs = fit_model(tg, draws=a.draws, tune=a.tune, chains=a.chains, seed=a.seed, cores=a.cores)
    import arviz as az
    sm = az.summary(idata, var_names=["a0", "sd_rw", "sd_team", "sd_trw", "sd_opp", "sd_ref", "sd_wk", "sd_game", "alpha", "c0", "beta"])
    ndiv = int(idata.sample_stats["diverging"].values.sum())
    print(f"[fit] {time.time() - t0:.0f}s | divergences {ndiv} | max r_hat {float(sm['r_hat'].max()):.3f} | min ess_bulk {float(sm['ess_bulk'].min()):.0f}")
    if ndiv or float(sm["r_hat"].max()) > 1.05:
        print("[fit] WARNING: sampler diagnostics look poor; consider --draws 1000 --tune 1000")
    home = games[(games.season >= cur - 2)]
    roof = home[home.roof.notna()].groupby("home_team").roof.apply(lambda x: float(x.isin(["dome", "closed"]).mean()))
    outdoor_wind = tg[(tg.dome == 0) & (tg.season >= 2018)].wind10.median() * 10.0
    meta = dict(teams=teams, refs=refs, active_refs=sorted(last[~last.referee.str.startswith("UNK_")].referee.unique()),
                n_seasons=int(tg.season.max() - tg.season.min() + 1), last_season=int(tg.season.max()), last_week=int(last.week.max()),
                n_rows=len(tg), covs=COVS, first_season=int(tg.season.min()), typical_wind=float(outdoor_wind),
                team_dome={t: int(roof.get(t, float(t in FIXED_ROOF)) > 0.5) for t in teams}, trained=time.strftime("%Y-%m-%d"),
                ref_games={r: int(n) for r, n in tg[~tg.referee.str.startswith("UNK_")].groupby("referee").game_id.nunique().items()},
                ref_last_season={r: int(x) for r, x in tg[~tg.referee.str.startswith("UNK_")].groupby("referee").season.max().items()},
                sampler=dict(draws=a.draws, tune=a.tune, chains=a.chains, seed=a.seed, since=a.since))
    out = Path(a.model_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "posterior.npz", **post)
    json.dump(meta, open(out / "meta.json", "w"))
    print(f"[done] saved {out}/posterior.npz and meta.json")


def main():
    ap = argparse.ArgumentParser(description="NFL penalty-flag model: predict flags in any game and price prop lines")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, alts):
        p.add_argument("--model-dir", default=str(MODEL_DIR))
        p.add_argument("--data-dir", default=str(DATA_DIR))
        p.add_argument("--hold", type=float, default=0.07, help="margin on the two-way price (assumption)")
        p.add_argument("--crew-shrink", type=float, default=0.6, help="scale on crew effects; raw backtest crew effects were ~2x too large")
        p.add_argument("--alts", type=int, default=alts, help="alternate lines either side of the main line")
        p.add_argument("--nrep", type=int, default=12, help="simulations per posterior draw")
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--offline", action="store_true", help="do not contact the network")

    p = sub.add_parser("predict", help="forecast and price ONE game (scheduled or hypothetical)")
    common(p, 0)
    p.add_argument("--game-id")
    p.add_argument("--home"); p.add_argument("--away"); p.add_argument("--week", type=int); p.add_argument("--season", type=int)
    p.add_argument("--referee"); p.add_argument("--temp", type=float); p.add_argument("--wind", type=float)
    p.add_argument("--dome", action="store_true"); p.add_argument("--outdoors", action="store_true"); p.add_argument("--primetime", action="store_true")
    p.add_argument("--line", type=float, help="price a specific book line"); p.add_argument("--odds", help="its American odds, e.g. -110/-110")
    p.add_argument("--market", default="game_thrown", choices=["game_thrown", "game_accepted", "home_thrown", "away_thrown", "home_accepted", "away_accepted"])
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_predict)

    p = sub.add_parser("week", help="price every unplayed game of a week and write a CSV")
    common(p, 2)
    p.add_argument("--week", type=int, required=True); p.add_argument("--season", type=int, default=None)
    p.add_argument("--referee", action="append", default=[], help='GAME_ID="Name" (repeatable)')
    p.add_argument("--referees-file", help="CSV with columns game_id, referee")
    p.add_argument("--md", help="also write a readable Markdown sheet (lines + crew effects)")
    p.add_argument("--include-played", action="store_true"); p.add_argument("--out", default="lines.csv")
    p.set_defaults(fn=cmd_week)

    p = sub.add_parser("actual", help="what actually happened: flags thrown / accepted in a game that has been played (optionally settle a lines CSV)")
    p.add_argument("--data-dir", default=str(DATA_DIR)); p.add_argument("--offline", action="store_true")
    p.add_argument("--game-id"); p.add_argument("--home"); p.add_argument("--away"); p.add_argument("--week", type=int); p.add_argument("--season", type=int)
    p.add_argument("--list", action="store_true", help="print every flag"); p.add_argument("--lines", help="a lines CSV from `week` to settle against")
    p.add_argument("--alts", action="store_true", help="also settle alternate lines"); p.add_argument("--show-lines", action="store_true", help="in slate mode, print every settled line"); p.add_argument("--refresh", action="store_true", help="re-download this season's play-by-play first")
    p.set_defaults(fn=cmd_actual)

    p = sub.add_parser("crews", help="show estimated referee-crew effects on the flag rate")
    p.add_argument("--model-dir", default=str(MODEL_DIR)); p.add_argument("--crew-shrink", type=float, default=0.6)
    p.add_argument("--all", action="store_true", help="include retired / inactive referees"); p.add_argument("--csv")
    p.set_defaults(fn=cmd_crews)

    p = sub.add_parser("update", help="download data, rebuild, refit (~15 min), save the model")
    p.add_argument("--model-dir", default=str(MODEL_DIR)); p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--draws", type=int, default=500); p.add_argument("--tune", type=int, default=500); p.add_argument("--chains", type=int, default=2)
    p.add_argument("--cores", type=int, default=None); p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--since", type=int, default=FIRST_SEASON, help="first season to train on (e.g. 2018: fits in ~2 min instead of ~13)")
    p.add_argument("--refresh-all", action="store_true", help="re-download every season's play-by-play")
    p.set_defaults(fn=cmd_update)

    argv, k = sys.argv[1:], 0                     # let "--odds -110/-110" work (argparse reads a leading '-' as a new flag)
    fixed = []
    while k < len(argv):
        if argv[k] == "--odds" and k + 1 < len(argv):
            fixed.append(f"--odds={argv[k + 1]}"); k += 2
        else:
            fixed.append(argv[k]); k += 1
    a = ap.parse_args(fixed)
    if a.cmd == "week" and a.season is None:
        a.season = load_model(a.model_dir)[0]["last_season"]
    warnings.filterwarnings("ignore")
    a.fn(a)


if __name__ == "__main__":
    main()
