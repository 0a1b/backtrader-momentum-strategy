from __future__ import annotations

"""
Live site for S5ov = S5 + seatbelts (the validated overlay from validate_dd85.py):

  S5ov = S5 (z(M1_100)+z(M1_250), top-2 inverse-vol, biweekly, 10 bps)
       + depth trigger : S5 portfolio value >15% below its high-water mark
                         -> 50% S5 / 25% GLD / 25% cash
       + velocity trigger: S5 fell >10% in the last 20 days
                         -> at least 50% risk-off (same split)

Same structure as build_site_s5.py (which it reuses for scraping, downloads and
the S5/S1/S0 ranking): scrape NDX constituents -> download prices (Yahoo,
auto-adjusted) -> recompute signals -> static dark site + CSV.

What is extra vs build_site_s5.py:
  * a standalone backfill of S5's own portfolio value over the downloaded window
    (biweekly, causal: decision at close d applied from day d+1, 10 bps per
    |delta weight|, cash until PIT-eligible) so the seatbelt state is real,
    * seatbelt state card: drawdown-from-high vs -15%, 20d velocity vs -10%,
    * recommended allocation (S5 / GLD / cash) and a NAV-vs-high-water sparkline,
  * honest caveat: the high-water mark is seeded at the START of the downloaded
    window (default 3y), not at 2015. For a longer memory set YF_PERIOD=5y or 10y.

Outputs: ranking.csv, s5ov_state.csv, index.html
Run: ../venv314/bin/python build_site_s5ov.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

from build_site_s5 import (
    _download_close,
    get_nasdaq100_symbols,
    get_price_data,
    build_ranking,
    _pick_rows,
    LB_SHORT,
    LB_LONG,
    MIN_HISTORY,
    DISPLAY_N,
    INVEST,
)

# --------------------------------------------------------------------------
# Configuration (mirrors the validated config; overridable via env)
# --------------------------------------------------------------------------

DD_TRIGGER_PCT = float(os.getenv("DD_TRIGGER_PCT", "15"))   # depth trigger: % below HWM
VEL_TRIGGER_PCT = float(os.getenv("VEL_TRIGGER_PCT", "10"))  # velocity trigger: % loss
VEL_WIN = int(os.getenv("VEL_WIN", "20"))                   #   ... in this many days
OFF_LEVEL = float(os.getenv("OFF_LEVEL", "0.5"))            # equity exposure when ON
GOLD_FRAC = float(os.getenv("GOLD_FRAC", "0.5"))            # of freed capital in gold
COST_BPS = float(os.getenv("COST_BPS", "10"))
TOP_N_S5 = 2
VOL_WIN = 20

HERE = Path(__file__).resolve().parent
GLD_CACHE = HERE / "s5ov_gld_cache.csv"
YF_PERIOD = os.getenv("YF_PERIOD", "3y")


# --------------------------------------------------------------------------
# Full-history study signals (mirror of btcore/signals.py, for the backfill)
# --------------------------------------------------------------------------

def _rolling_log_slope(logp: pd.DataFrame, lb: int) -> pd.DataFrame:
    """Trailing OLS slope of log-price, per column (identical to btcore)."""
    S1 = logp.rolling(lb).sum()
    cnt = logp.rolling(lb).count()
    x = np.arange(lb, dtype=float)
    x_bar = x.mean()
    den = float(np.sum((x - x_bar) ** 2))
    Sxy = logp.mul(0.0)
    for i in range(logp.shape[1]):
        y = logp.iloc[:, i].to_numpy()
        y = np.where(np.isnan(y), 0.0, y)
        Sxy.iloc[lb - 1:, i] = np.convolve(y, x[::-1], mode="valid")
    slope = (Sxy - x_bar * S1) / den
    slope[cnt != lb] = np.nan
    return slope


def momentum_m1(close: pd.DataFrame, lb: int) -> pd.DataFrame:
    r = ((close / close.shift(lb) - 1.0) * 100.0).clip(lower=0.0)
    gate = np.minimum(1.0, np.ceil(r))
    slope = _rolling_log_slope(np.log(close), lb)
    score = gate * np.power(1.0 + slope, 252)
    return score.where(slope.notna())


def blend_z(close: pd.DataFrame, lbs: tuple = (100, 250)) -> pd.DataFrame:
    out = None
    for lb in lbs:
        s = momentum_m1(close, lb)
        m = s.mean(axis=1).to_numpy()[:, None]
        sd = s.std(axis=1).to_numpy()[:, None]
        sd = np.where((sd == 0) | np.isnan(sd), np.nan, sd)
        z = (s - m) / sd
        out = z if out is None else out + z
    return out


def realized_vol(close: pd.DataFrame, w: int = VOL_WIN) -> pd.DataFrame:
    r = np.log(close / close.shift(1))
    return r.rolling(w).std() * np.sqrt(252.0)


# --------------------------------------------------------------------------
# S5 backfill (close-to-close approximation of the study engine)
# --------------------------------------------------------------------------

def simulate_s5_backfill(close: pd.DataFrame) -> pd.DataFrame:
    """Backfill S5's own NAV over the downloaded window.

    Causal: decision at close of biweekly Monday d, new weights applied to
    returns from day d+1 (close-to-close here), 10 bps one-way per |Δw|,
    cash until a stock has >= MIN_HISTORY closes.
    """
    idx = close.index
    c = close.to_numpy(dtype=float)
    T, N = c.shape
    sig = blend_z(close, (LB_SHORT, LB_LONG))
    vol = realized_vol(close)
    hist = close.notna().sum().to_numpy()
    sigv = sig.to_numpy(dtype=float)
    volv = vol.to_numpy(dtype=float)
    rets = np.full((T, N), np.nan)
    rets[1:] = c[1:] / c[:-1] - 1.0

    mons = idx[idx.weekday == 0]
    biw = mons[::2]
    pos = {d: i for i, d in enumerate(idx)}
    exec_at: dict[int, int] = {}
    for d in biw:
        i = pos.get(d)
        if i is not None and i + 1 < T:
            exec_at[i + 1] = i

    nav = np.full(T, np.nan)
    nav[0] = 1.0
    w = np.zeros(N)
    for t in range(1, T):
        if t in exec_at:
            d = exec_at[t]
            elig = hist >= MIN_HISTORY
            srow = np.where(elig, sigv[d], np.nan)
            vrow = volv[d]
            ok = np.isfinite(srow) & np.isfinite(vrow) & (vrow > 0)
            picks = np.where(ok)[0]
            nw = np.zeros(N)
            if len(picks) >= TOP_N_S5:
                order = picks[np.argsort(srow[picks])[::-1]][:TOP_N_S5]
                vw = 1.0 / vrow[order]
                nw[order] = (vw / vw.sum()) * INVEST
            turn = float(np.abs(nw - w).sum())
            cost = turn * COST_BPS / 1e4
            w = nw
        else:
            cost = 0.0
        r_t = rets[t]
        held = np.where(np.isfinite(r_t) & (w > 0), w * np.where(np.isfinite(r_t), r_t, 0.0), 0.0)
        nav[t] = nav[t - 1] * (1.0 + held.sum() - cost)

    nav = pd.Series(nav, index=idx)
    hwm = nav.cummax()
    depth_dd = nav / hwm - 1.0
    vel = nav / nav.shift(VEL_WIN) - 1.0
    depth_on = depth_dd < -DD_TRIGGER_PCT / 100.0
    vel_on = vel < -VEL_TRIGGER_PCT / 100.0
    exposure = np.where(depth_on | vel_on, OFF_LEVEL, 1.0)
    return pd.DataFrame({
        "nav": nav, "hwm": hwm,
        "depth_dd_pct": depth_dd * 100.0,
        f"vel{VEL_WIN}_pct": vel * 100.0,
        "depth_on": depth_on, "vel_on": vel_on,
        "exposure": exposure,
    })


def seatbelt_state(back: pd.DataFrame) -> dict:
    last = back.dropna(subset=["nav"]).iloc[-1]
    on = bool(last["depth_on"] or last["vel_on"])
    equity = float(last["exposure"])
    gold = (1.0 - equity) * GOLD_FRAC
    return {
        "asof": str(last.name.date()),
        "on": on,
        "active": "depth" if bool(last["depth_on"]) else ("velocity" if bool(last["vel_on"]) else None),
        "depth_on": bool(last["depth_on"]),
        "vel_on": bool(last["vel_on"]),
        "depth_dd_pct": float(last["depth_dd_pct"]),
        "vel_pct": float(last[f"vel{VEL_WIN}_pct"]),
        "exposure": equity,
        "equity_pct": equity * 100.0,
        "gold_pct": gold * 100.0,
        "cash_pct": (1.0 - equity - gold) * 100.0,
    }


def get_gld() -> pd.Series | None:
    try:
        df = _download_close(["GLD"], YF_PERIOD, max_retries=3)
        s = df["GLD"] if "GLD" in df.columns else df.iloc[:, 0]
        s = s.dropna()
        if s.empty:
            raise RuntimeError("empty GLD download")
        s.to_csv(GLD_CACHE, header=["GLD"])
        return s
    except Exception as e:
        if GLD_CACHE.exists():
            print(f"GLD download failed ({e}); using cache.")
            return pd.read_csv(GLD_CACHE, index_col=0, parse_dates=True).iloc[:, 0]
        print(f"GLD unavailable ({e}); gold shown as 0%.")
        return None


# --------------------------------------------------------------------------
# Site
# --------------------------------------------------------------------------

def _spark_svg(back: pd.DataFrame, w: int = 800, h: int = 190) -> str:
    v = back.dropna(subset=["nav"])
    if len(v) < 10:
        return ""
    nav, hwm = v["nav"], v["hwm"]
    lo, hi = min(v["nav"].min(), v["hwm"].min()), max(nav.max(), hwm.max())
    span = (hi - lo) or 1.0
    pad_l, pad_r, pad_t, pad_b = 8, 8, 10, 22
    xs = np.linspace(pad_l, w - pad_r, len(v))

    def pts(s: pd.Series) -> str:
        ys = [h - pad_b - (x - lo) / span * (h - pad_t - pad_b) for x in s.to_numpy()]
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))

    def runs(mask: pd.Series) -> list[str]:
        """One point-string per contiguous True run (so gaps stay gaps)."""
        out, cur = [], []
        for i, m in enumerate(mask.to_numpy()):
            if m:
                cur.append(f"{xs[i]:.1f},{h - pad_b - (nav.iloc[i] - lo) / span * (h - pad_t - pad_b):.1f}")
            else:
                if cur:
                    out.append(" ".join(cur))
                    cur = []
        if cur:
            out.append(" ".join(cur))
        return out

    on_mask = (v["depth_on"] | v["vel_on"])
    on_polys = "\n".join(
        f'<polyline points="{p}" fill="none" stroke="#f87171" stroke-width="2"/>'
        for p in runs(on_mask)
    )
    return f"""
    <svg viewBox="0 0 {w} {h}" style="width:100%;height:auto;display:block">
      <polyline points="{pts(hwm)}" fill="none" stroke="#475569" stroke-width="1.5"
        stroke-dasharray="5 4"/>
      {on_polys}
      <polyline points="{pts(nav)}" fill="none" stroke="#60a5fa" stroke-width="2"/>
      <text x="10" y="14" fill="#94a3b8" font-size="11">S5 backfilled value (solid) vs
        high-water mark (dashed); red = seatbelt ON</text>
      <text x="{w - 190}" y="{h - 6}" fill="#94a3b8" font-size="11">
        {v.index[0].date()} … {v.index[-1].date()}
      </text>
      <text x="10" y="{h - 6}" fill="#94a3b8" font-size="11">
        min {lo:.2f} — max {hi:.2f} (start = 1.00)
      </text>
    </svg>"""


def _alloc_bar(st: dict) -> str:
    eq, g, c = st["equity_pct"], st["gold_pct"], st["cash_pct"]
    def seg(pct, color, label):
        if pct < 0.05:
            return ""
        return (f'<div class="seg" style="width:{pct:.1f}%;background:{color}" '
                f'title="{label} {pct:.0f}%"><span>{label} {pct:.0f}%</span></div>')
    return (f'<div class="alloc">'
            f'{seg(eq, "#60a5fa", "S5")}{seg(g, "#fbbf24", "GLD")}{seg(c, "#334155", "Cash")}'
            f"</div>")


def render_html(ranked, meta, back, st, gld) -> str:
    rows = []
    w_s5 = meta["weights_s5"]
    for t, r in ranked.iterrows():
        rank = int(r["rank"])
        badge = "pick" if rank <= TOP_N_S5 else ""
        w = w_s5.get(t, np.nan)
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

    # effective S5ov exposure per pick
    eff_rows = []
    for t in meta["picks_s5"]:
        w = w_s5.get(t, 0.0)
        eff_rows.append(
            f'<tr><td class="ticker">{t}</td>'
            f'<td class="num">{w * 100:.1f}%</td>'
            f'<td class="num wt">{w * st["exposure"] * 100:.1f}%</td></tr>'
        )
    eff_rows.append(f'<tr><td class="ticker">GLD</td><td class="num">—</td>'
                    f'<td class="num wt">{st["gold_pct"]:.1f}%</td></tr>')
    eff_rows.append(f'<tr><td class="ticker">Cash</td><td class="num">—</td>'
                    f'<td class="num wt">{st["cash_pct"]:.1f}%</td></tr>')
    eff_html = "\n".join(eff_rows)

    status_cls = "on" if st["on"] else "off"
    status_txt = (f"BELTS ON — {st['active']} trigger · "
                  f"{st['equity_pct']:.0f}% S5 / {st['gold_pct']:.0f}% GLD / "
                  f"{st['cash_pct']:.0f}% cash") if st["on"] else "BELTS OFF — 100% in S5"
    dd_col = "#f87171" if st["depth_on"] else "#34d399"
    vel_col = "#f87171" if st["vel_on"] else "#34d399"
    gld_px = f"${gld.iloc[-1]:.2f}" if gld is not None and len(gld) else "n/a"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NDX Momentum — S5ov (S5 + seatbelts)</title>
  <style>
    :root {{ --bg: #0b0f17; --card: rgba(17,24,39,0.88); --text: #e5e7eb; --muted: #94a3b8; --line: #243041; --accent: #60a5fa; --gold: #fbbf24; --green: #34d399; --red: #f87171; }}
    body {{ margin: 0; padding: 24px; font-family: system-ui; background: var(--bg); color: var(--text); display: flex; justify-content: center; }}
    .wrap {{ width: 100%; max-width: 860px; }}
    h1 {{ font-size: 20px; }}
    .card {{ margin-top: 14px; background: var(--card); border: 1px solid rgba(36,48,65,0.9); border-radius: 14px; overflow: hidden; }}
    .bar {{ display:flex; flex-wrap: wrap; gap: 6px; justify-content: space-between; padding: 14px 16px; border-bottom: 1px solid var(--line); }}
    .pill {{ font-size: 12px; color: var(--muted); padding: 6px 10px; border: 1px solid var(--line); border-radius: 999px; background: rgba(2,6,23,0.35); }}
    .pill b {{ color: var(--text); }}
    .status {{ padding: 16px; font-size: 17px; font-weight: 800; }}
    .status.on {{ color: var(--red); border-bottom: 1px solid var(--line); }}
    .status.off {{ color: var(--green); border-bottom: 1px solid var(--line); }}
    .trig {{ display:flex; justify-content: space-between; gap: 10px; padding: 10px 16px; border-bottom: 1px solid var(--line); font-size: 14px; }}
    .trig:last-of-type {{ border-bottom: none; }}
    .trig .val {{ font-weight: 700; }}
    .alloc {{ display:flex; height: 26px; border-radius: 8px; overflow: hidden; margin: 14px 16px; border: 1px solid var(--line); }}
    .seg {{ color: #0b1220; font-size: 11px; font-weight: 800; display:flex; align-items:center; justify-content:center; overflow:hidden; white-space:nowrap; }}
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
    h2 {{ font-size: 14px; color: var(--muted); text-transform: uppercase; padding: 14px 16px 0; margin: 0; letter-spacing: 0.04em; }}
    .spark {{ padding: 10px 16px 4px; }}
    .footer {{ margin-top: 12px; color: var(--muted); font-size: 12px; line-height: 1.6; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>NDX Momentum — S5ov (S5 + seatbelts)</h1>
    <div class="card">
      <div class="bar">
        <div class="pill">Score: <b>z(M1<sub>{LB_SHORT}</sub>) + z(M1<sub>{LB_LONG}</sub>)</b></div>
        <div class="pill">Top-2 <b>inverse-vol</b></div>
        <div class="pill">Triggers: <b>&gt;{DD_TRIGGER_PCT:.0f}% below high</b> or <b>&gt;{VEL_TRIGGER_PCT:.0f}% in {VEL_WIN}d</b></div>
        <div class="pill">As of: <b>{meta['asof']}</b></div>
      </div>
      <div class="status {status_cls}">{status_txt}</div>
      <div class="trig">
        <span>Depth trigger — S5 value vs its high:
          <span class="val" style="color:{dd_col}">{st['depth_dd_pct']:+.1f}%</span>
          (fires at −{DD_TRIGGER_PCT:.0f}%)
        </span>
        <span>{'ON' if st['depth_on'] else 'off'}</span>
      </div>
      <div class="trig">
        <span>Velocity trigger — S5 {VEL_WIN}d return:
          <span class="val" style="color:{vel_col}">{st['vel_pct']:+.1f}%</span>
          (fires at −{VEL_TRIGGER_PCT:.0f}%)
        </span>
        <span>{'ON' if st['vel_on'] else 'off'}</span>
      </div>
      <div class="trig"><span>GLD price</span><span>{gld_px}</span></div>
      <div class="trig"><span>Recommended allocation</span><span></span></div>
      {_alloc_bar(st)}
      <div class="spark">{_spark_svg(back)}</div>
    </div>

    <div class="card">
      <h2 style="padding-top:14px">S5ov exposure (weight × belt state)</h2>
      <table>
        <thead><tr><th>Position</th><th class="num">S5 weight</th><th class="num">S5ov exposure</th></tr></thead>
        <tbody>{eff_html}</tbody>
      </table>
    </div>

    <div class="card">
      <h2 style="padding-top:14px">Current positions per study</h2>
      <table>
        <tbody>{picks_html}</tbody>
      </table>
    </div>

    <div class="card">
      <h2 style="padding-top:14px">S5 ranking (top {min(DISPLAY_N, len(ranked))})</h2>
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

    <div class="footer">
      Universe: Wikipedia NASDAQ-100 constituents · Prices: Yahoo Finance (auto-adjusted) ·
      {meta['n_universe']} eligible of {meta['n_rows']} rows ending {meta['asof']}.<br/>
      S5ov = S5 + seatbelts (validated in <code>validate_dd85.py</code>): when S5's own value is
      &gt;{DD_TRIGGER_PCT:.0f}% below its high-water mark, or it has fallen &gt;{VEL_TRIGGER_PCT:.0f}% in {VEL_WIN} days,
      equity exposure drops to {OFF_LEVEL * 100:.0f}% and the freed capital is split
      {GOLD_FRAC * 100:.0f}% GLD / {100 - GOLD_FRAC * 100:.0f}% cash.<br/>
      <b>Caveats:</b> the seatbelt state is computed from a {YF_PERIOD} backfill of S5 (close-to-close
      approximation of the study engine, 10 bps), so the high-water mark is seeded at the start of
      that window, not in 2015 — set <code>YF_PERIOD=5y</code>/10y for a longer memory. Backtest is a
      2020s-boom property (validated DD reduction is robust, CAGR edge over S0 is not 95%-significant);
      this page is a live snapshot, not a recommendation.
    </div>
  </div>
</body>
</html>
"""


def main() -> int:
    symbols = get_nasdaq100_symbols()
    close = get_price_data(symbols)
    gld = get_gld()
    if gld is not None:
        gld = gld.reindex(close.index).ffill()

    ranked, meta = build_ranking(close)

    back = simulate_s5_backfill(close)
    st = seatbelt_state(back)

    out = ranked.reset_index()
    out = out.rename(columns={out.columns[0]: "ticker"})
    out["s5_weight_pct"] = out["ticker"].map(lambda t: meta["weights_s5"].get(t, np.nan) * 100)
    out.to_csv(HERE / "ranking.csv", index=False)

    back.drop(columns=["depth_on", "vel_on"]).round(4).to_csv(HERE / "s5ov_state.csv")

    html = render_html(ranked.head(DISPLAY_N), meta, back, st, gld)
    (HERE / "index.html").write_text(html, encoding="utf-8")

    print(f"\nS5ov as of {st['asof']}  (eligible: {meta['n_universe']}, GLD: "
          f"{gld.iloc[-1] if gld is not None and gld.notna().any() else 'n/a'})")
    print(f"  depth:  {st['depth_dd_pct']:+.1f}% vs -{DD_TRIGGER_PCT:.0f}%  -> "
          f"{'ON' if st['depth_on'] else 'off'}")
    print(f"  velocity: {st['vel_pct']:+.1f}%/{VEL_WIN}d vs -{VEL_TRIGGER_PCT:.0f}%  -> "
          f"{'ON' if st['vel_on'] else 'off'}")
    print(f"  allocation: S5 {st['equity_pct']:.0f}% / GLD {st['gold_pct']:.0f}% / "
          f"Cash {st['cash_pct']:.0f}%")
    print("  S5 picks: " + ", ".join(
        f"{t} {meta['weights_s5'][t] * 100:.1f}% (x{st['exposure']:.2f})"
        for t in meta["picks_s5"]))
    print("\nWrote ranking.csv, s5ov_state.csv, index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
