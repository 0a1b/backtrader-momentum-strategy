from __future__ import annotations

import os
import datetime as dt
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import yfinance as yf


WIKI_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"

LOOKBACK_PERIOD = int(os.getenv("LOOKBACK_PERIOD", "250"))
ROC_PERIOD = int(os.getenv("ROC_PERIOD", "250"))
TOP_N = int(os.getenv("TOP_N", "5"))
YF_PERIOD = os.getenv("YF_PERIOD", "2y")  # ensure enough data

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def get_nasdaq100_symbols() -> list[str]:
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(WIKI_URL, headers=headers, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    tables = soup.find_all("table", class_="wikitable")
    target = None
    for table in tables:
        ths = [th.get_text(strip=True) for th in table.find_all("th")]
        if "Ticker" in ths:
            target = table
            break
    if target is None:
        raise RuntimeError("Could not find NASDAQ-100 components table (Ticker column).")

    symbols = []
    for row in target.find_all("tr")[1:]:
        cols = row.find_all(["td", "th"])
        if not cols:
            continue
        ticker = cols[0].get_text(strip=True)
        # Yahoo uses BRK-B style; Wikipedia sometimes uses dots
        ticker = ticker.replace(".", "-")
        symbols.append(ticker)

    # De-dup, preserve order
    out, seen = [], set()
    for s in symbols:
        if s and s not in seen:
            out.append(s)
            seen.add(s)
    return out


def _download_close(tickers: list[str], period: str) -> pd.DataFrame:
    df = yf.download(
        tickers,
        period=period,
        interval="1d",
        auto_adjust=True,
        group_by="column",
        threads=True,
        progress=False,
    )

    if isinstance(df.columns, pd.MultiIndex):
        close = df["Close"].copy()
    else:
        # single-ticker fallback
        close = df[["Close"]].rename(columns={"Close": tickers[0]})

    close.index = pd.to_datetime(close.index)
    close = close.sort_index()
    return close


def momentum(data: np.ndarray, lookback_period: int, roc_period: int) -> np.ndarray:
    """
    This function mirrors the notebook's momentum(...) implementation:
    - ROC gating: ceil(((shifted - data) / data) * 100) clipped to [0,1]
    - Rolling regression slope on log(price) over lookback_period
    - Annualize: (1 + slope) ** 252
    - Final: res = roc * values   (correlation is computed but NOT applied)
    """
    shifted_values = np.zeros(data.shape)
    shifted_values[: shifted_values.shape[0] - roc_period, :] = data[roc_period:, :]

    roc = ((shifted_values - data) / data) * 100
    roc = roc[: roc.shape[0] - roc_period, :]
    roc = np.ceil(roc)
    roc = np.maximum(np.full(roc.shape, 0), roc)
    roc = np.minimum(np.full(roc.shape, 1), roc)

    xxx = np.vstack([np.arange(lookback_period), np.ones(lookback_period)]).T
    # precompute (X'X)^-1 X'
    beta = np.linalg.inv(xxx.T @ xxx) @ xxx.T  # shape (2, lookback_period)

    values = np.zeros((data.shape[0] - lookback_period, data.shape[1]))
    correl = np.zeros((data.shape[0] - lookback_period, data.shape[1]))  # unused later (kept for fidelity)

    for i in range(data.shape[1]):
        cur_data = np.log(data[:, i])

        view = np.lib.stride_tricks.sliding_window_view(cur_data, (lookback_period,))[1:, :]
        lin_reg = beta @ view.T  # shape (2, n_windows)

        roll_mat = lin_reg[0]  # slope
        lin_reg_b = lin_reg[1]  # intercept

        # Build fitted line and correlation (not used in final res, but kept)
        x = view
        line = np.arange(x.shape[1])
        m = roll_mat[:, np.newaxis]
        n = lin_reg_b[:, np.newaxis]
        y = (m * line) + n

        centered_x = x - np.mean(x, axis=1, keepdims=True)
        centered_y = y - np.mean(y, axis=1, keepdims=True)
        cov_xy = np.mean(centered_x * centered_y, axis=1)
        var_x = np.mean(centered_x**2, axis=1)
        var_y = np.mean(centered_y**2, axis=1)
        corrcoef_xy = cov_xy / (np.sqrt(var_x * var_y))

        correl[:, i] = corrcoef_xy
        values[:, i] = roll_mat

    values = 1 + values
    values = np.power(values, 252)

    if roc.shape[0] - values.shape[0] >= 0:
        new_shape = roc.shape[0] - values.shape[0]
        roc = roc[new_shape:, :]
    else:
        new_shape = values.shape[0] - roc.shape[0]
        values = values[new_shape:, :]
        correl = correl[new_shape:, :]

    # Final calculation (as in notebook; correlation NOT applied)
    res = roc * values  # * correl

    original = np.full((data.shape[0], data.shape[1]), np.nan)
    original[original.shape[0] - res.shape[0] :, :] = res
    return original


def render_html(ranked: pd.DataFrame, updated: str, lookback: int, roc: int) -> str:
    rows = []
    for _, r in ranked.iterrows():
        rank = int(r["rank"])
        ticker = r["ticker"]
        mom = float(r["momentum"])
        badge = "top" if rank <= 10 else ""
        rows.append(
            f"""
            <tr class="{badge}">
              <td class="rank">#{rank}</td>
              <td class="ticker">{ticker}</td>
              <td class="num mom">{mom:.6f}</td>
            </tr>
            """
        )

    rows_html = "\n".join(rows)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NASDAQ-100 Momentum</title>
  <style>
    :root {{
      --bg: #0b0f17;
      --card: rgba(17,24,39,0.88);
      --text: #e5e7eb;
      --muted: #94a3b8;
      --line: #243041;
      --accent: #60a5fa;
      --gold: #fbbf24;
    }}
    body {{
      margin: 0; padding: 24px;
      font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      background:
        radial-gradient(1200px 800px at 20% 0%, rgba(96,165,250,0.16), transparent 60%),
        radial-gradient(1000px 700px at 100% 30%, rgba(251,191,36,0.12), transparent 55%),
        var(--bg);
      color: var(--text);
      display: flex;
      justify-content: center;
    }}
    .wrap {{ width: 100%; max-width: 820px; }}
    h1 {{ margin: 0; font-size: 22px; letter-spacing: 0.2px; }}
    .sub {{ margin-top: 6px; color: var(--muted); font-size: 13px; line-height: 1.35; }}
    .card {{
      margin-top: 14px;
      background: var(--card);
      border: 1px solid rgba(36,48,65,0.9);
      border-radius: 14px;
      overflow: hidden;
      box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    }}
    .bar {{
      display:flex; justify-content: space-between; align-items:center;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      gap: 10px;
      flex-wrap: wrap;
    }}
    .pill {{
      font-size: 12px; color: var(--muted);
      padding: 6px 10px; border: 1px solid var(--line); border-radius: 999px;
      background: rgba(2,6,23,0.35);
    }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 12px 14px; border-bottom: 1px solid var(--line); }}
    th {{
      font-size: 11px; color: var(--muted);
      text-transform: uppercase; letter-spacing: 0.08em;
      text-align: left;
      background: rgba(2,6,23,0.35);
    }}
    td.num, th.num {{ text-align: right; }}
    td.rank {{ width: 76px; color: var(--muted); font-weight: 800; }}
    td.ticker {{ font-weight: 850; letter-spacing: 0.02em; }}
    td.mom {{
      color: var(--gold);
      font-variant-numeric: tabular-nums;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    }}
    tr.top td.rank {{ color: var(--accent); }}
    tr.top td.ticker {{ color: #fff; }}
    .footer {{
      margin-top: 12px;
      color: var(--muted);
      font-size: 12px;
      display: flex; justify-content: space-between; flex-wrap: wrap; gap: 10px;
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div>
      <h1>NASDAQ‑100 Momentum Ranking</h1>
      <div class="sub">Calculated like your notebook’s <code>momentum(...)</code>: ROC gate × annualized slope. Top 10 highlighted.</div>
    </div>

    <div class="card">
      <div class="bar">
        <div class="pill">Lookback: {lookback} days</div>
        <div class="pill">ROC period: {roc} days</div>
        <div class="pill">Updated: {updated}</div>
      </div>

      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Ticker</th>
            <th class="num">Momentum</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>

    <div class="footer">
      <div><a href="ranking.csv">Download CSV</a></div>
      <div>Universe source: Wikipedia NASDAQ‑100 constituents.</div>
    </div>
  </div>
</body>
</html>
"""


def main() -> int:
    symbols = get_nasdaq100_symbols()

    # Use QQQ as benchmark calendar (similar spirit to the notebook's benchmark handling)
    bench = _download_close(["QQQ"], YF_PERIOD)["QQQ"].dropna()
    target_index = bench.index

    close = _download_close(symbols, YF_PERIOD)
    close = close.reindex(target_index)

    # Keep only symbols with complete history over the benchmark calendar
    close = close.dropna(axis=1, how="any")

    # Safety: drop non-positive prices (log)
    close = close.loc[:, (close > 0).all(axis=0)]

    if close.shape[1] < 10:
        raise RuntimeError(f"Too few symbols after filtering for full history: {close.shape[1]}")

    # Momentum calc (notebook-equivalent)
    mom = momentum(close.to_numpy(dtype=float), LOOKBACK_PERIOD, ROC_PERIOD)
    mom_df = pd.DataFrame(mom, index=close.index, columns=close.columns)

    last = mom_df.iloc[-1].dropna()
    ranked = (
        last.sort_values(ascending=False)
        .head(TOP_N)
        .reset_index()
        .rename(columns={"index": "ticker", last.name: "momentum"})
    )
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))

    ranked.to_csv("ranking.csv", index=False)

    updated = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    html = render_html(ranked, updated, LOOKBACK_PERIOD, ROC_PERIOD)
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
