from __future__ import annotations

"""
Live ranking site for the study's strategies (S0 / S1 / S5).

Same idea as https://github.com/0a1b/backtrader-momentum-strategy/blob/main/build_site.py
(scrape NDX constituents -> download prices -> rank -> static site), but the
scores are the *study* signals from `btcore/signals.py`, recomputed here
standalone so this file runs anywhere:

  S5  score = z(M1_100) + z(M1_250)        top-2, inverse-vol weights  (study winner)
  S1  score = M1_250                       top-2, inverse-vol weights
  S0  score = M1_250                       top-1                        (base)

  M1(lb) = 1{ROC_lb > 0} * (1 + OLS_slope_logP_lb)^252   (the original notebook formula)
  z()    = cross-sectional z-score on the same date (causal)

Execution model this site assumes (from the study, see WALK_FORWARD_STUDY.md):
  decision at close, fill at next rebalance open (every other Monday),
  one-way 10 bps, 100% invested, PIT-eligible (>= 270 closes of history).

Outputs: ranking.csv (full S5 ranking) and index.html (dark static site).
"""

import json
import os
import re
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf

try:
    from curl_cffi import requests as cffi_requests
    _HAS_CURL_CFFI = True
except ImportError:
    _HAS_CURL_CFFI = False


# --------------------------------------------------------------------------
# Configuration (mirrors the study's pre-registered setup)
# --------------------------------------------------------------------------

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"
LB_SHORT = int(os.getenv("LB_SHORT", "100"))       # S5 short lookback
LB_LONG = int(os.getenv("LB_LONG", "250"))         # S5 long lookback / S0 & S1 lookback
MIN_HISTORY = int(os.getenv("MIN_HISTORY", "270")) # PIT eligibility, as in the study
VOL_WIN = 20                                       # inverse-vol window (study: 20d)
TOP_N_S5 = 2                                       # S5 / S1 picks
TOP_N_S0 = 1                                       # S0 pick
DISPLAY_N = int(os.getenv("DISPLAY_N", "10"))      # rows shown in the HTML ranking
YF_PERIOD = os.getenv("YF_PERIOD", "3y")
INVEST = 1.0                                       # engine default: fully invested

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

HERE = Path(__file__).resolve().parent
SYMBOLS_CACHE = HERE / "nasdaq100_symbols.json"
PRICE_CACHE = HERE / "s5_close_cache.csv"

MIN_ROWS_REQUIRED = max(LB_SHORT, LB_LONG) + 30


# --------------------------------------------------------------------------
# NASDAQ-100 constituent scraping (same approach as the reference repo)
# --------------------------------------------------------------------------

def _fetch_nasdaq100_html() -> str:
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
    r = requests.get(WIKI_URL, headers=headers, timeout=30)
    r.raise_for_status()
    if not r.text:
        raise RuntimeError("Wikipedia returned an empty page.")
    return r.text


TICKER_RE = re.compile(r"^[A-Z]{1,6}([.-][A-Z])?$")


def _looks_like_components_table(table) -> tuple[bool, int, float]:
    rows = table.find_all("tr")
    if len(rows) < 90:
        return False, len(rows), 0.0

    checked = 0
    matches = 0
    for row in rows[1:111]:
        cells = row.find_all(["td", "th"])
        if not cells:
            continue
        first = cells[0].get_text(strip=True).replace(".", "-")
        checked += 1
        if TICKER_RE.match(first):
            matches += 1

    if checked == 0:
        return False, len(rows), 0.0

    ratio = matches / checked
    return ratio > 0.85, len(rows), ratio


def _find_components_table(soup, tables):
    for table in tables:
        ths = [th.get_text(strip=True) for th in table.find_all("th")]
        if any(h.strip().lower() == "ticker" for h in ths):
            is_match, _, _ = _looks_like_components_table(table)
            if is_match:
                return table

    candidates = []
    for table in soup.find_all("table"):
        is_match, row_count, ratio = _looks_like_components_table(table)
        if is_match:
            candidates.append((row_count, ratio, table))

    if candidates:
        candidates.sort(key=lambda c: c[0], reverse=True)
        return candidates[0][2]

    return None


def _scrape_nasdaq100_symbols() -> list[str]:
    html = _fetch_nasdaq100_html()
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table", class_="wikitable")

    target = _find_components_table(soup, tables)
    if target is None:
        debug_path = HERE / "wiki_debug.html"
        debug_path.write_text(html, encoding="utf-8")
        raise RuntimeError(
            f"Could not find NASDAQ-100 table at {WIKI_URL}. "
            f"Dumped response to {debug_path} for inspection."
        )

    symbols = []
    for row in target.find_all("tr")[1:]:
        cols = row.find_all(["td", "th"])
        if not cols:
            continue
        ticker = cols[0].get_text(strip=True).replace(".", "-")
        symbols.append(ticker)

    out, seen = [], set()
    for s in symbols:
        if s and s not in seen:
            out.append(s)
            seen.add(s)

    if len(out) < 90:
        raise RuntimeError(f"Only parsed {len(out)} symbols, expected ~100 — table likely malformed.")

    return out


def get_nasdaq100_symbols() -> list[str]:
    try:
        symbols = _scrape_nasdaq100_symbols()
        SYMBOLS_CACHE.write_text(json.dumps(symbols))
        print(f"Scraped {len(symbols)} NASDAQ-100 symbols from Wikipedia.")
        return symbols
    except Exception as e:
        if SYMBOLS_CACHE.exists():
            cached = json.loads(SYMBOLS_CACHE.read_text())
            print(f"Wikipedia scrape failed ({e}); falling back to cached list "
                  f"({len(cached)} symbols).")
            return cached
        raise RuntimeError(
            "Wikipedia scrape failed and no cached symbol list exists yet."
        ) from e


# --------------------------------------------------------------------------
# Price downloads (browser impersonation, retry/backoff, cache fallback)
# --------------------------------------------------------------------------

def _make_yf_session():
    if _HAS_CURL_CFFI:
        return cffi_requests.Session(impersonate="chrome")
    return None


def _download_close(tickers: list[str], period: str, max_retries: int = 5) -> pd.DataFrame:
    session = _make_yf_session()
    last_err = None

    for attempt in range(max_retries):
        try:
            kwargs = dict(
                period=period,
                interval="1d",
                auto_adjust=True,
                group_by="column",
                threads=False,
                progress=False,
            )
            if session is not None:
                kwargs["session"] = session

            df = yf.download(tickers, **kwargs)

            if df is not None and not df.empty:
                break
        except Exception as e:
            last_err = e
            df = None

        wait = min(2 ** attempt * 10, 120)
        print(f"[{attempt + 1}/{max_retries}] Download empty/failed; retrying in {wait}s...")
        time.sleep(wait)
    else:
        raise RuntimeError(f"yfinance download failed after {max_retries} attempts: {last_err}")

    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"].copy()
    else:
        close = df[["Close"]].rename(columns={"Close": tickers[0]})

    close.index = pd.to_datetime(close.index)
    close = close.sort_index()
    return close


def get_price_data(symbols: list[str]) -> pd.DataFrame:
    try:
        bench = _download_close(["QQQ"], YF_PERIOD)["QQQ"].dropna()
        if len(bench) < MIN_ROWS_REQUIRED:
            raise RuntimeError(
                f"QQQ benchmark only has {len(bench)} rows — likely rate-limited."
            )
        target_index = bench.index

        close = _download_close(symbols, YF_PERIOD)
        close = close.reindex(target_index)
        close = close.loc[:, (close > 0).all(axis=0) & close.notna().all(axis=0)]

        if close.shape[1] < 10:
            raise RuntimeError(f"Too few symbols after filtering: {close.shape[1]}")

        close.to_csv(PRICE_CACHE)
        print(f"Downloaded fresh price data: {close.shape[0]} rows x {close.shape[1]} symbols.")
        return close

    except Exception as e:
        if PRICE_CACHE.exists():
            print(f"Live price download failed ({e}); falling back to cached data.")
            close = pd.read_csv(PRICE_CACHE, index_col=0, parse_dates=True)
            if close.shape[0] < MIN_ROWS_REQUIRED or close.shape[1] < 10:
                raise RuntimeError("Cached price data insufficient.") from e
            return close
        raise RuntimeError(f"Live price download failed ({e}) and no cached data.") from e


# --------------------------------------------------------------------------
# Study signals — standalone re-implementation of btcore/signals.py (last row)
# --------------------------------------------------------------------------

def _rolling_log_slope_last(logp: pd.DataFrame, lb: int) -> np.ndarray:
    """OLS slope of log-price over the last `lb` days, per column.

    NaN where the window contains any NaN. Matches btcore._rolling_log_slope
    on the final row (verified against the study engine).
    """
    x = np.arange(lb, dtype=float)
    x_bar = x.mean()
    den = float(np.sum((x - x_bar) ** 2))
    out = np.full(logp.shape[1], np.nan)
    for i in range(logp.shape[1]):
        v = logp.iloc[-lb:, i].to_numpy(dtype=float)
        if np.isnan(v).any():
            continue
        Sxy = float(np.sum(x * v))
        S1 = float(np.sum(v))
        out[i] = (Sxy - x_bar * S1) / den
    return out


def momentum_m1_last(close: pd.DataFrame, lb: int) -> pd.Series:
    """M1 momentum at the last row, exact study formula:

        score = 1{ROC_lb > 0} * (1 + OLS_slope_logP_lb)^252

    (binary ceil/clip gate quirk preserved — it is part of the tested signal.)
    """
    r = (close.iloc[-1] / close.shift(lb).iloc[-1] - 1.0) * 100.0
    r = r.clip(lower=0.0)
    gate = np.minimum(1.0, np.ceil(r))
    slope = _rolling_log_slope_last(np.log(close), lb)
    score = pd.Series(np.asarray(gate, dtype=float), index=close.columns)
    score = score * np.power(1.0 + slope, 252)
    score[~np.isfinite(slope)] = np.nan
    return score


def zscore_row(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score (pandas defaults: skipna, ddof=1) — as in btcore.blend_z."""
    m = s.mean()
    sd = s.std()
    if sd is None or np.isnan(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - m) / sd


def realized_vol_last(close: pd.DataFrame, w: int = VOL_WIN) -> pd.Series:
    """Annualized trailing-w daily log-return std, at the last row (study vol for inv-vol)."""
    r = np.log(close / close.shift(1))
    return r.iloc[-w:].std() * np.sqrt(252.0)


def inv_vol_weights(picks: list[str], vol: pd.Series, invest: float = INVEST) -> dict[str, float]:
    """Target weights for the picked tickers, proportional to 1/vol (study engine rule:
    a pick with missing/non-positive vol gets 0 and the rest is renormalized)."""
    v = np.array([vol.get(t, np.nan) for t in picks], dtype=float)
    ok = np.isfinite(v) & (v > 0)
    w = np.where(ok, 1.0 / np.where(ok, v, 1.0), 0.0)
    s = w.sum()
    w = w / s * invest if s > 0 else np.zeros_like(w)
    return dict(zip(picks, w))


# --------------------------------------------------------------------------
# Ranking + site
# --------------------------------------------------------------------------

def build_ranking(close: pd.DataFrame) -> pd.DataFrame:
    m_short = momentum_m1_last(close, LB_SHORT)
    m_long = momentum_m1_last(close, LB_LONG)
    z_short = zscore_row(m_short)
    z_long = zscore_row(m_long)
    s5 = z_short + z_long
    vol = realized_vol_last(close)

    hist = close.notna().sum()
    eligible = (hist >= MIN_HISTORY) & s5.notna()

    rank = pd.DataFrame({
        "m1_short": m_short,
        "m1_long": m_long,
        "z_short": z_short,
        "z_long": z_long,
        "s5": s5,
        "vol20": vol,
        "hist_days": hist,
        "eligible": eligible,
    })

    ranked = rank[rank["eligible"]].sort_values("s5", ascending=False)
    ranked["rank"] = np.arange(1, len(ranked) + 1)

    # target weights for the strategies
    picks_s5 = list(ranked.index[:TOP_N_S5])
    # S1 uses the same top-N rule but ranks by M1_long alone
    s1_rank = rank[rank["eligible"]].sort_values("m1_long", ascending=False)
    picks_s1 = list(s1_rank.index[:TOP_N_S5])
    picks_s0 = list(s1_rank.index[:TOP_N_S0])

    w_s5 = inv_vol_weights(picks_s5, vol)
    w_s1 = inv_vol_weights(picks_s1, vol)
    w_s0 = dict(zip(picks_s0, [INVEST]))

    meta = {
        "asof": str(close.index[-1].date()),
        "n_rows": len(close),
        "n_universe": int(rank["eligible"].sum()),
        "picks_s5": picks_s5, "weights_s5": w_s5,
        "picks_s1": picks_s1, "weights_s1": w_s1,
        "picks_s0": picks_s0, "weights_s0": w_s0,
    }
    return ranked, meta


def _pick_rows(meta) -> str:
    rows = []
    for label, key in (("S5 — z(M1₁₀₀)+z(M1₂₅₀), top-2 inv-vol", "s5"),
                       ("S1 — M1₂₅₀, top-2 inv-vol", "s1"),
                       ("S0 — M1₂₅₀, top-1 (base)", "s0")):
        picks = meta[f"picks_{key}"]
        weights = meta[f"weights_{key}"]
        cells = " · ".join(
            f"<b>{t}</b> {weights[t] * 100:.1f}%" for t in picks
        ) or "—"
        rows.append(
            f'<tr><td class="label">{label}</td><td class="picks">{cells}</td></tr>'
        )
    return "\n".join(rows)


def render_html(ranked: pd.DataFrame, meta: dict) -> str:
    rows = []
    for t, r in ranked.iterrows():
        rank = int(r["rank"])
        badge = "pick" if rank <= TOP_N_S5 else ""
        w = meta["weights_s5"].get(t, np.nan)
        wcell = f"{w * 100:.1f}%" if np.isfinite(w) else ""
        rows.append(
            f"""
            <tr class="{badge}">
              <td class="rank">#{rank}</td>
              <td class="ticker">{t}</td>
              <td class="num">{r['z_short']:.2f}</td>
              <td class="num">{r['z_long']:.2f}</td>
              <td class="num mom">{r['s5']:.3f}</td>
              <td class="num">{r['vol20'] * 100:.1f}%</td>
              <td class="num wt">{wcell}</td>
            </tr>
            """
        )
    rows_html = "\n".join(rows)
    picks_html = _pick_rows(meta)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NDX Momentum — S5 (study)</title>
  <style>
    :root {{ --bg: #0b0f17; --card: rgba(17,24,39,0.88); --text: #e5e7eb; --muted: #94a3b8; --line: #243041; --accent: #60a5fa; --gold: #fbbf24; --green: #34d399; }}
    body {{ margin: 0; padding: 24px; font-family: system-ui; background: var(--bg); color: var(--text); display: flex; justify-content: center; }}
    .wrap {{ width: 100%; max-width: 860px; }}
    .card {{ margin-top: 14px; background: var(--card); border: 1px solid rgba(36,48,65,0.9); border-radius: 14px; overflow: hidden; }}
    .bar {{ display:flex; flex-wrap: wrap; gap: 6px; justify-content: space-between; padding: 14px 16px; border-bottom: 1px solid var(--line); }}
    .pill {{ font-size: 12px; color: var(--muted); padding: 6px 10px; border: 1px solid var(--line); border-radius: 999px; background: rgba(2,6,23,0.35); }}
    .pill b {{ color: var(--text); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 11px 14px; border-bottom: 1px solid var(--line); }}
    th {{ font-size: 11px; color: var(--muted); text-transform: uppercase; text-align: left; background: rgba(2,6,23,0.35); }}
    td.num, th.num {{ text-align: right; }}
    td.rank {{ color: var(--muted); font-weight: 800; }}
    td.ticker {{ font-weight: 850; }}
    td.mom {{ color: var(--gold); }}
    td.wt {{ color: var(--green); font-weight: 700; }}
    td.label {{ font-size: 13px; color: var(--muted); }}
    td.picks {{ font-size: 15px; }}
    tr.pick td {{ background: rgba(96,165,250,0.08); }}
    tr.pick td.rank {{ color: var(--accent); }}
    .footer {{ margin-top: 12px; color: var(--muted); font-size: 12px; line-height: 1.6; }}
    h2 {{ font-size: 14px; color: var(--muted); text-transform: uppercase; padding: 14px 16px 0; margin: 0; letter-spacing: 0.04em; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>NDX Momentum — Study Strategies (S5 / S1 / S0)</h1>
    <div class="card">
      <div class="bar">
        <div class="pill">Score: <b>z(M1<sub>{LB_SHORT}</sub>) + z(M1<sub>{LB_LONG}</sub>)</b></div>
        <div class="pill">Top-2 <b>inverse-vol</b></div>
        <div class="pill">Min hist: <b>{MIN_HISTORY} d</b></div>
        <div class="pill">As of: <b>{meta['asof']}</b></div>
      </div>
      <table>
        <thead><tr>
          <th>Rank</th><th>Ticker</th>
          <th class="num">z(M1 {LB_SHORT})</th>
          <th class="num">z(M1 {LB_LONG})</th>
          <th class="num">S5 score</th>
          <th class="num">Vol 20d</th>
          <th class="num">S5 weight</th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>

    <div class="card">
      <h2 style="padding-top:14px">Current positions per study</h2>
      <table>
        <tbody>{picks_html}</tbody>
      </table>
    </div>

    <div class="footer">
      Universe: Wikipedia NASDAQ-100 constituents · Prices: Yahoo Finance (auto-adjusted) ·
      {meta['n_universe']} eligible of {meta['n_rows']} rows ending {meta['asof']}.<br/>
      Assumes the study execution model: decision at close, fill at the next rebalance
      <b>open</b> (every other Monday), one-way 10 bps, 100% invested, point-in-time eligible.<br/>
      Signal and parameters as pre-registered in <code>WALK_FORWARD_STUDY.md</code> /
      <code>run_study.py</code> — the backtest is a 2020s-boom property with ~57% historical
      max DD; this page is a live snapshot, not a recommendation.
    </div>
  </div>
</body>
</html>
"""


def main() -> int:
    symbols = get_nasdaq100_symbols()
    close = get_price_data(symbols)

    ranked, meta = build_ranking(close)

    out = ranked.reset_index()
    out = out.rename(columns={out.columns[0]: "ticker"})  # index may carry a name
    out["s5_weight_pct"] = out["ticker"].map(
        lambda t: meta["weights_s5"].get(t, np.nan) * 100
    )
    out.to_csv(HERE / "ranking.csv", index=False)

    html = render_html(ranked.head(DISPLAY_N).assign(rank=ranked.head(DISPLAY_N)["rank"]), meta)
    (HERE / "index.html").write_text(html, encoding="utf-8")

    print(f"\nS5 as of {meta['asof']} (eligible: {meta['n_universe']}):")
    print(f"  S5 top-2 inv-vol : " +
          ", ".join(f"{t} {w * 100:.1f}%" for t, w in zip(meta["picks_s5"], meta["weights_s5"].values())))
    print(f"  S1 top-2 inv-vol : " +
          ", ".join(f"{t} {w * 100:.1f}%" for t, w in zip(meta["picks_s1"], meta["weights_s1"].values())))
    print(f"  S0 top-1         : " +
          ", ".join(f"{t} {w * 100:.1f}%" for t, w in zip(meta["picks_s0"], meta["weights_s0"].values())))
    print("\nWrote ranking.csv and index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
