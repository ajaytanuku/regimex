# RegimeX: Analytics-Driven Study of Indian Stock Market Regimes

A Streamlit web app that analyses how the Indian stock market behaves in four regimes
(Bull, Sideways, Bear, Panic), which factors drive Nifty 50 stocks in each regime, and
whether regime-aware models build a better portfolio.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The site opens at http://localhost:8501. Python 3.10 to 3.12 is recommended.

- **Live NSE data (Yahoo Finance)** downloads real daily data for Nifty 50 stocks, the Nifty 50 index and India VIX.
- **Demo data (offline)** generates a synthetic market so the app works without internet (useful for a demo if the network fails).

If the live download fails, the app switches to demo data and shows a warning.

## Pages

| Page | What it shows | Techniques |
|---|---|---|
| Home | Current regime, Nifty and VIX, regime timeline | Summary |
| Live market | Nifty 50 and India VIX ticking, 1-minute intraday chart, today's movers, watchlist (auto-refresh every 10 to 60 s) | Live Yahoo Finance quotes |
| Paper trading | Rs 10 lakh of virtual money: buy/sell at live prices, holdings with live P&L, trade history, one-click rebalance to the RegimeX portfolio | SQLite order book |
| 1. Data & quality | Missing-value heatmap, quality report, cleaned CSV download | Cleaning, outlier flags |
| 2. Exploratory analysis | Return distribution, volatility, sector rotation, calendar and Budget-day effects, FII/DII flows | EDA |
| 3. Statistical tests | Hypothesis tests, correlation matrix, ADF, seasonal decomposition, GARCH | Jarque-Bera, Granger, t-test, ANOVA, ADF, GARCH(1,1) |
| 4. Market regimes | Regime timeline, profile, transition matrix, correlation and sector returns by regime | Gaussian Hidden Markov Model |
| 5. Clustering & PCA | Stock clusters, silhouette scores, PCA factors, dendrogram | K-Means, hierarchical clustering, PCA |
| 6. Prediction & portfolio | Global vs regime-aware models, feature importance, backtest, current portfolio | LightGBM, mean-variance optimisation, walk-forward backtest |
| 7. Key insights | Findings written as sentences for the report | Summary |

## Optional FII/DII data

Download daily FII/DII trading activity from the NSE website and save it as a CSV with these columns
(net values in Rs crore), then upload it in the sidebar:

```csv
Date,FII_Net,DII_Net
02-01-2025,-1234.5,2100.3
```

The app then adds a Granger causality test (do FII flows predict Nifty?) and a lead-lag chart.

## Method notes

- **Regimes:** a 4-state HMM on the 21-day Nifty trend, 21-day realised volatility and India VIX. The highest-volatility state is Panic; the rest are ordered by trend into Bear, Sideways and Bull.
- **No look-ahead in the backtest:** the HMM and the models are trained only on data before the backtest start date. Regimes in the test period are labelled using only past data, and the last 5 training days are purged because their targets overlap the test period.
- **Portfolio:** every 5 trading days, the top N stocks by predicted 5-day return are weighted by mean-variance optimisation (long-only, weight cap). Exposure is cut to 75% in Bear and 50% in Panic. Transaction costs are deducted on turnover.
- **Delivery percentage:** Yahoo Finance has no delivery data, so the volume-spike test uses volume as a proxy.

## Project structure

```
app.py            Streamlit website (all pages)
src/data.py       Data download, demo data, cleaning
src/analytics.py  EDA, statistical tests, HMM regimes, clustering, PCA
src/predict.py    Features, LightGBM models, portfolio optimisation, backtest
src/live.py       Live quotes, market status, paper-trading account and rebalancing
```

## Live prices and paper trading

- Prices come from Yahoo Finance and refresh automatically. NSE trades 9:15 am to 3:30 pm IST, Monday to Friday; outside those hours the page shows the last close.
- Paper trading uses virtual money only. Each trader name has its own account. Orders fill at the latest price, with 0.1% charges.
- "Rebalance to RegimeX portfolio" places the orders needed to match the model's current recommended weights.
- In demo mode the live page uses a simulated ticking feed, so it moves even offline.

## Put it online (free, about 5 minutes)

1. Create a new public repository on GitHub (for example `regimex`) and upload everything in this folder (`app.py`, `src/`, `requirements.txt`, `packages.txt`, `README.md`, `.gitignore`).
2. Go to https://share.streamlit.io, sign in with GitHub, and click **Create app**.
3. Pick the repository, branch `main`, main file `app.py`. Under **Advanced settings** choose Python 3.12.
4. Click **Deploy**. The app gets a public link like `https://regimex.streamlit.app`.

Notes for the online version: Yahoo Finance sometimes rate-limits shared cloud servers. If that happens the app falls back to demo data and shows a warning, so wait a few minutes or click "Reload daily data". Paper trades are stored on the server and reset when the app restarts.

Educational project. Not investment advice.
