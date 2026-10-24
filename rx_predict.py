"""Phase 5: regime-aware forecasting, portfolio optimisation and backtest."""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from rx_analytics import REGIMES, RegimeModel, regime_features
from rx_data import INDEX, VIX, short

HORIZON = 5  # predict 5-day forward return
FEATURES = ["r1", "r5", "r21", "vol21", "rsi14", "dist_ma50", "vol_z", "mkt_r5", "vix", "vix_chg5"]


def _rsi(p, n=14):
    d = p.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def build_panel(close, volume, stocks):
    mkt_r5 = close[INDEX].pct_change(5)
    vix = close[VIX]
    vix_chg5 = vix.pct_change(5)
    frames = []
    for s in stocks:
        p, v = close[s], np.log(volume[s].replace(0, np.nan))
        r = p.pct_change()
        f = pd.DataFrame({
            "r1": r, "r5": p.pct_change(5), "r21": p.pct_change(21),
            "vol21": r.rolling(21).std() * np.sqrt(252),
            "rsi14": _rsi(p),
            "dist_ma50": p / p.rolling(50).mean() - 1,
            "vol_z": (v - v.rolling(60).mean()) / v.rolling(60).std(),
            "mkt_r5": mkt_r5, "vix": vix, "vix_chg5": vix_chg5,
            "target": p.shift(-HORIZON) / p - 1,
        })
        f["stock"] = s
        frames.append(f)
    panel = pd.concat(frames).dropna(subset=FEATURES)
    panel.index.name = "date"
    return panel


def _lgbm():
    try:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=31,
                             min_child_samples=100, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, random_state=0, verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, random_state=0)


def _importance(m):
    if hasattr(m, "booster_"):
        imp = m.booster_.feature_importance("gain")
        return pd.Series(imp / imp.sum() * 100, index=FEATURES)
    return pd.Series(np.nan, index=FEATURES)


def _metrics(df, col):
    d = df.dropna(subset=["target"])
    acc = (np.sign(d[col]) == np.sign(d["target"])).mean() * 100
    ic = d.groupby(level=0).apply(lambda g: stats.spearmanr(g[col], g["target"])[0]
                                  if len(g) > 3 else np.nan).mean()
    rmse = np.sqrt(((d[col] - d["target"]) ** 2).mean()) * 100
    return {"Directional accuracy (%)": acc, "Information coefficient": ic, "RMSE (%)": rmse}


def run_models(close, volume, stocks, test_start):
    """Train a single global model and per-regime models; evaluate on the test period."""
    test_start = pd.Timestamp(test_start)
    X = regime_features(close)
    # fit regimes on the training period only, label test days without look-ahead
    rm = RegimeModel().fit(X[X.index < test_start])
    reg = pd.concat([rm.label(X[X.index < test_start]), rm.label_filtered(X, test_start)])

    panel = build_panel(close, volume, stocks)
    panel["regime"] = reg.reindex(panel.index).values
    panel = panel.dropna(subset=["regime"])
    # last HORIZON days of training have targets overlapping the test period -> purge
    cutoff = panel.index[panel.index < test_start].unique().sort_values()[-HORIZON]
    train = panel[(panel.index < cutoff) & panel["target"].notna()]
    test = panel[panel.index >= test_start].copy()

    glob = _lgbm().fit(train[FEATURES], train["target"])
    test["pred_global"] = glob.predict(test[FEATURES])

    importances = {"Global": _importance(glob)}
    test["pred_regime"] = test["pred_global"]
    models = {}
    for g in REGIMES:
        tr = train[train["regime"] == g]
        if len(tr) < 500:
            continue
        m = _lgbm().fit(tr[FEATURES], tr["target"])
        models[g] = m
        importances[g] = _importance(m)
        idx = test["regime"] == g
        if idx.any():
            test.loc[idx, "pred_regime"] = m.predict(test.loc[idx, FEATURES])

    metrics = pd.DataFrame({"Single global model": _metrics(test, "pred_global"),
                            "Regime-aware models": _metrics(test, "pred_regime")}).T
    return {"test": test, "metrics": metrics, "importance": pd.DataFrame(importances),
            "regimes": reg, "regime_kind": rm.kind, "models": models, "global": glob}


def optimise(mu, cov, cap=0.25, gamma=10.0):
    """Mean-variance: maximise mu'w - gamma/2 w'Sw, long-only, weights <= cap."""
    n = len(mu)
    cap = max(cap, 1.0 / n)
    res = minimize(lambda w: -(w @ mu - gamma / 2 * w @ cov @ w), np.ones(n) / n,
                   bounds=[(0, cap)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], method="SLSQP")
    w = np.clip(res.x if res.success else np.ones(n) / n, 0, None)
    return w / w.sum()


EXPOSURE = {"Bull": 1.0, "Sideways": 1.0, "Bear": 0.75, "Panic": 0.5}


def backtest(close, test, stocks, top_n=8, cap=0.25, cost_bps=10, col="pred_regime", risk_control=True):
    r = close[stocks].pct_change()
    dates = test.index.unique().sort_values()
    rebal = dates[::HORIZON]
    w_prev = pd.Series(0.0, index=stocks)
    daily, history = [], []
    for i, d in enumerate(rebal):
        snap = test.loc[[d]].set_index("stock")
        pick = snap[col].nlargest(top_n)
        hist = r.loc[:d].iloc[-120:][pick.index].dropna()
        cov = hist.cov().values * HORIZON
        w = optimise(pick.values, cov, cap)
        regime = snap["regime"].iloc[0]
        expo = EXPOSURE.get(regime, 1.0) if risk_control else 1.0
        w_new = pd.Series(0.0, index=stocks)
        w_new[pick.index] = w * expo
        turnover = (w_new - w_prev).abs().sum()
        end = rebal[i + 1] if i + 1 < len(rebal) else dates[-1]
        period = r.loc[(r.index > d) & (r.index <= end), stocks].fillna(0)
        pr = period @ w_new
        if len(pr):
            pr.iloc[0] -= turnover * cost_bps / 1e4
        daily.append(pr)
        history.append({"date": d, "regime": regime, "exposure": expo, "turnover": turnover})
        w_prev = w_new
    strat = pd.concat(daily)
    nifty = close[INDEX].pct_change().reindex(strat.index).fillna(0)
    ew = r.reindex(strat.index)[stocks].mean(axis=1).fillna(0)
    rets = pd.DataFrame({"RegimeX strategy": strat, "Nifty 50 buy-and-hold": nifty, "Equal-weight stocks": ew})
    return rets, pd.DataFrame(history).set_index("date")


def perf(rets, rf=0.06):
    out = {}
    for c in rets:
        x = rets[c]
        eq = (1 + x).cumprod()
        yrs = len(x) / 252
        cagr = eq.iloc[-1] ** (1 / yrs) - 1
        vol = x.std() * np.sqrt(252)
        out[c] = {"Total return (%)": (eq.iloc[-1] - 1) * 100, "CAGR (%)": cagr * 100,
                  "Volatility (%)": vol * 100, "Sharpe ratio": (cagr - rf) / vol,
                  "Max drawdown (%)": ((eq / eq.cummax()) - 1).min() * 100}
    return pd.DataFrame(out).T


def current_portfolio(close, volume, stocks, res, top_n=8, cap=0.25):
    """Portfolio for the latest available day."""
    panel = build_panel(close, volume, stocks)
    last = panel.index.max()
    snap = panel.loc[[last]].set_index("stock")
    regime = res["regimes"].iloc[-1]
    model = res["models"].get(regime, res["global"])
    snap["pred"] = model.predict(snap[FEATURES])
    pick = snap["pred"].nlargest(top_n)
    cov = close[pick.index].pct_change().iloc[-120:].cov().values * HORIZON
    w = optimise(pick.values, cov, cap) * EXPOSURE.get(regime, 1.0)
    port = pd.DataFrame({"Stock": [short(s) for s in pick.index],
                         "Predicted 5-day return (%)": pick.values * 100,
                         "Weight (%)": w * 100})
    return last, regime, port
