"""Phase 1: data collection and cleaning."""
import numpy as np
import pandas as pd

# Nifty 50 constituents (Yahoo Finance tickers) and their sectors
STOCKS = {
    "RELIANCE.NS": "Energy", "ONGC.NS": "Energy",
    "TCS.NS": "IT", "INFY.NS": "IT", "HCLTECH.NS": "IT", "WIPRO.NS": "IT", "TECHM.NS": "IT",
    "HDFCBANK.NS": "Banking", "ICICIBANK.NS": "Banking", "SBIN.NS": "Banking",
    "KOTAKBANK.NS": "Banking", "AXISBANK.NS": "Banking", "INDUSINDBK.NS": "Banking",
    "BAJFINANCE.NS": "Financial Services", "BAJAJFINSV.NS": "Financial Services",
    "HINDUNILVR.NS": "FMCG", "ITC.NS": "FMCG", "NESTLEIND.NS": "FMCG", "BRITANNIA.NS": "FMCG",
    "MARUTI.NS": "Auto", "M&M.NS": "Auto", "BAJAJ-AUTO.NS": "Auto", "EICHERMOT.NS": "Auto",
    "SUNPHARMA.NS": "Pharma", "DRREDDY.NS": "Pharma", "CIPLA.NS": "Pharma",
    "LT.NS": "Infrastructure", "ULTRACEMCO.NS": "Cement",
    "TATASTEEL.NS": "Metals", "JSWSTEEL.NS": "Metals", "HINDALCO.NS": "Metals",
    "NTPC.NS": "Power", "POWERGRID.NS": "Power",
    "BHARTIARTL.NS": "Telecom", "ASIANPAINT.NS": "Consumer", "TITAN.NS": "Consumer",
}
INDEX = "^NSEI"      # Nifty 50
VIX = "^INDIAVIX"    # India VIX


def short(t):
    return t.replace(".NS", "")


def load_live(tickers, start):
    """Download adjusted prices and volumes from Yahoo Finance."""
    import yfinance as yf
    raw = yf.download(list(tickers) + [INDEX, VIX], start=start,
                      auto_adjust=True, progress=False, threads=True)
    if raw is None or raw.empty:
        raise RuntimeError("No data returned from Yahoo Finance")
    close, volume = raw["Close"], raw["Volume"]
    if close[INDEX].dropna().empty:
        raise RuntimeError("Nifty index data missing")
    return close, volume


def load_demo(tickers, start, seed=7):
    """Synthetic regime-switching market, so the app works offline."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, pd.Timestamp.today().normalize())
    n = len(dates)
    # regimes: 0 bull, 1 sideways, 2 bear, 3 panic
    mu = np.array([0.0009, 0.0001, -0.0007, -0.0025])
    sig = np.array([0.008, 0.007, 0.012, 0.030])
    P = np.array([[0.985, 0.010, 0.004, 0.001],
                  [0.012, 0.978, 0.008, 0.002],
                  [0.006, 0.010, 0.976, 0.008],
                  [0.030, 0.010, 0.040, 0.920]])
    reg = np.zeros(n, dtype=int)
    for i in range(1, n):
        reg[i] = rng.choice(4, p=P[reg[i - 1]])
    mkt = rng.normal(mu[reg], sig[reg])
    nifty = 8000 * np.exp(np.cumsum(mkt))
    vix = np.clip(12 + 900 * sig[reg] + rng.normal(0, 1.5, n), 9, 85)

    close, volume = {}, {}
    sectors = sorted(set(STOCKS[t] for t in tickers))
    sec_shock = {s: rng.normal(0, 0.006, n) for s in sectors}
    for t in tickers:
        beta = rng.uniform(0.6, 1.4)
        alpha = rng.normal(0.0002, 0.0002)
        r = alpha + beta * mkt + sec_shock[STOCKS[t]] + rng.normal(0, 0.011, n)
        p = rng.uniform(200, 3000) * np.exp(np.cumsum(r))
        p[rng.random(n) < 0.003] = np.nan            # missing values to clean
        close[t] = p
        base = rng.uniform(1e6, 1e7)
        volume[t] = base * np.exp(rng.normal(0, 0.35, n) + 8 * np.abs(r))
    close[INDEX], close[VIX] = nifty, vix
    volume[INDEX] = volume[VIX] = np.nan
    return pd.DataFrame(close, index=dates), pd.DataFrame(volume, index=dates)


def clean(close, volume, tickers, max_missing=0.2):
    """Return cleaned frames plus a data-quality report."""
    close = close.sort_index()
    close = close[~close.index.duplicated()]
    stocks = [t for t in tickers if t in close.columns]
    before = close[stocks].isna().mean()

    # keep only days when the index traded (NSE trading days)
    close = close[close[INDEX].notna()]
    volume = volume.reindex(close.index)

    # drop tickers with too much missing history
    miss = close[stocks].isna().mean()
    dropped = miss[miss > max_missing].index.tolist()
    stocks = [t for t in stocks if t not in dropped]

    close[stocks + [VIX]] = close[stocks + [VIX]].ffill(limit=5)
    close = close.dropna(subset=stocks)
    volume = volume.reindex(close.index)[stocks].ffill(limit=5).fillna(0)

    rets = close[stocks].pct_change()
    z = (rets - rets.mean()) / rets.std()
    outliers = (z.abs() > 6).sum()

    report = pd.DataFrame({
        "Missing % (raw)": (before * 100).round(2),
        "Missing % (clean)": (close[stocks].isna().mean() * 100).round(2).reindex(before.index),
        "Extreme return days (|z|>6)": outliers.reindex(before.index),
        "Status": ["Dropped" if t in dropped else "Kept" for t in before.index],
    })
    report.index = [short(t) for t in report.index]
    return close, volume, stocks, report


def fii_csv(file):
    """Optional FII/DII flows CSV: Date, FII_Net, DII_Net (Rs crore)."""
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    return df[[c for c in ["FII_Net", "DII_Net"] if c in df.columns]].astype(float)
