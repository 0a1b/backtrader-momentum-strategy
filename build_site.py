from __future__ import annotations

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


# Updated URL constant
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies"

LOOKBACK_PERIOD = int(os.getenv("LOOKBACK_PERIOD", "250"))
ROC_PERIOD = int(os.getenv("ROC_PERIOD", "250"))
TOP_N = int(os.getenv("TOP_N", "5"))
YF_PERIOD = os.getenv("YF_PERIOD", "2y")  # ensure enough data

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

SYMBOLS_CACHE = Path("nasdaq100_symbols.json")
PRICE_CACHE = Path("close_cache.csv")

MIN_ROWS_REQUIRED = LOOKBACK_PERIOD + ROC_PERIOD + 10


# --------------------------------------------------------------------------
# NASDAQ-100 constituent scraping (direct page fetch)
# --------------------------------------------------------------------------

def _fetch_nasdaq100_html() -> str:
    """
    Fetch the raw HTML of the provided NASDAQ-100 Wikipedia URL.
    """
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
    r = requests.get(WIKI_URL, headers=headers, timeout=30)
    r.raise_for_status()

    # Save to debug path on failure
    if not r.text:
        raise RuntimeError("Wikipedia returned an empty page.")

    return r.text


TICKER_RE = re.compile(r"^[A-Z]{1,6}([.-][A-Z])?$")


def _looks_like_components_table(table) -> tuple[bool, int, float]:
    """
    Structural check: does this table's first column look like ~100 stock
    tickers? Returns (is_match, row_count, ticker_ratio).
    """
    rows = table.find_all("tr")
    if len(rows) < 90:
        return False, len(rows), 0.0

    checked = 0
    matches = 0
    for row in rows[1:111]:  # skip header row, sample up to 110 data rows
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
    """
    Try the cheap header-text match first; fall back to a structural scan
    of every <table> on the page.
    """
    for table in tables:
        ths = [th.get_text(strip=True) for th in table.find_all("th")]
        if any(h.strip().lower() == "ticker" for h in ths):
            is_match, _, _ = _looks_like_components_table(table)
            if is_match:
                return table

    # Structural fallback: scan ALL tables on the page, any class.
    candidates = []
    for table in soup.find_all("table"):
        is_match, row_count, ratio = _looks_like_components_table(table)
        if is_match:
            candidates.append((row_count, ratio, table))

    if candidates:
        # Prefer the largest matching table
        candidates.sort(key=lambda c: c[0], reverse=True)
        return candidates[0][2]

    return None


def _scrape_nasdaq100_symbols() -> list[str]:
    html = _fetch_nasdaq100_html()
    soup = BeautifulSoup(html, "html.parser")
    # Finding wikitables specifically
    tables = soup.find_all("table", class_="wikitable")

    target = _find_components_table(soup, tables)

    if target is None:
        debug_path = Path("wiki_debug.html")
        debug_path.write_text(html, encoding="utf-8")
        all_headers = [[th.get_text(strip=True) for th in t.find_all("th")] for t in tables]
        headers_preview = " | ".join(str(h[:6]) for h in all_headers)
        raise RuntimeError(
            f"Could not find NASDAQ-100 table at {WIKI_URL}. "
            f"Dumped response to {debug_path} for inspection."
        )

    symbols = []
    for row in target.find_all("tr")[1:]:
        cols = row.find_all(["td", "th"])
        if not cols:
            continue
        ticker = cols[0].get_text(strip=True)
        ticker = ticker.replace(".", "-")  # Yahoo uses BRK-B style
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
            f"Wikipedia scrape failed ({e}) and no cached symbol list exists yet."
        ) from e


# --------------------------------------------------------------------------
# Price downloads (with browser impersonation, retry/backoff, cache fallback)
# --------------------------------------------------------------------------

def _make_yf_session():
    if _HAS_CURL_CFFI:
        return cffi_requests.Session(impersonate="chrome")
    return None  # yfinance will use its own default session


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
        close = close.dropna(axis=1, how="any")
        close = close.loc[:, (close > 0).all(axis=0)]

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
# Momentum calc
# --------------------------------------------------------------------------

def momentum(data: np.ndarray, lookback_period: int, roc_period: int) -> np.ndarray:
    if data.shape[0] < lookback_period + roc_period:
        raise ValueError("Not enough rows for requested periods.")

    shifted_values = np.zeros(data.shape)
    shifted_values[: shifted_values.shape[0] - roc_period, :] = data[roc_period:, :]

    roc = ((shifted_values - data) / data) * 100
    roc = roc[: roc.shape[0] - roc_period, :]
    roc = np.ceil(roc)
    roc = np.maximum(np.full(roc.shape, 0), roc)
    roc = np.minimum(np.full(roc.shape, 1), roc)

    xxx = np.vstack([np.arange(lookback_period), np.ones(lookback_period)]).T
    beta = np.linalg.inv(xxx.T @ xxx) @ xxx.T

    values = np.zeros((data.shape[0] - lookback_period, data.shape[1]))
    correl = np.zeros((data.shape[0] - lookback_period, data.shape[1]))

    for i in range(data.shape[1]):
        cur_data = np.log(data[:, i])
        view = np.lib.stride_tricks.sliding_window_view(cur_data, (lookback_period,))[1:, :]
        lin_reg = beta @ view.T

        roll_mat = lin_reg[0]
        lin_reg_b = lin_reg[1]

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

    res = roc * values
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
    :root {{ --bg: #0b0f17; --card: rgba(17,24,39,0.88); --text: #e5e7eb; --muted: #94a3b8; --line: #243041; --accent: #60a5fa; --gold: #fbbf24; }}
    body {{ margin: 0; padding: 24px; font-family: system-ui; background: var(--bg); color: var(--text); display: flex; justify-content: center; }}
    .wrap {{ width: 100%; max-width: 820px; }}
    .card {{ margin-top: 14px; background: var(--card); border: 1px solid rgba(36,48,65,0.9); border-radius: 14px; overflow: hidden; }}
    .bar {{ display:flex; justify-content: space-between; padding: 14px 16px; border-bottom: 1px solid var(--line); }}
    .pill {{ font-size: 12px; color: var(--muted); padding: 6px 10px; border: 1px solid var(--line); border-radius: 999px; background: rgba(2,6,23,0.35); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 12px 14px; border-bottom: 1px solid var(--line); }}
    th {{ font-size: 11px; color: var(--muted); text-transform: uppercase; text-align: left; background: rgba(2,6,23,0.35); }}
    td.num, th.num {{ text-align: right; }}
    td.rank {{ color: var(--muted); font-weight: 800; }}
    td.ticker {{ font-weight: 850; }}
    td.mom {{ color: var(--gold); }}
    tr.top td.rank {{ color: var(--accent); }}
    .footer {{ margin-top: 12px; color: var(--muted); font-size: 12px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>NASDAQ‑100 Momentum Ranking</h1>
    <div class="card">
      <div class="bar">
        <div class="pill">Lookback: {lookback} days</div>
        <div class="pill">ROC period: {roc} days</div>
        <div class="pill">Updated: {updated}</div>
      </div>
      <table>
        <thead><tr><th>Rank</th><th>Ticker</th><th class="num">Momentum</th></tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    <div class="footer">Universe source: Wikipedia NASDAQ‑100 constituents.</div>
  </div>
</body>
</html>
"""


def main() -> int:
    symbols = get_nasdaq100_symbols()
    close = get_price_data(symbols)

    mom = momentum(close.to_numpy(dtype=float), LOOKBACK_PERIOD, ROC_PERIOD)
    mom_df = pd.DataFrame(mom, index=close.index, columns=close.columns)

    last = mom_df.iloc[-1].dropna()

    ranked = (
        last.sort_values(ascending=False)
            .head(TOP_N)
            .reset_index(name="momentum")
    )
    ranked = ranked.rename(columns={ranked.columns[0]: "ticker"})
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))

    ranked.to_csv("ranking.csv", index=False)

    updated = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    html = render_html(ranked, updated, LOOKBACK_PERIOD, ROC_PERIOD)
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
