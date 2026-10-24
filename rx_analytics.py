"""Phases 2-4: EDA, statistical tests, regime detection, clustering and PCA."""
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from rx_data import INDEX, STOCKS, VIX, short

warnings.filterwarnings("ignore")

REGIMES = ["Bull", "Sideways", "Bear", "Panic"]
REGIME_COLORS = {"Bull": "#2e9e5b", "Sideways": "#8a8f98", "Bear": "#d9822b", "Panic": "#d1453b"}

EVENTS = {
    "2016-11-09": "Demonetisation",
    "2020-03-23": "COVID crash",
    "2022-05-04": "RBI surprise hike",
    "2024-06-04": "Election results",
}
BUDGET_DAYS = ["2016-02-29", "2017-02-01", "2018-02-01", "2019-02-01", "2019-07-05",
               "2020-02-01", "2021-02-01", "2022-02-01", "2023-02-01", "2024-02-01",
               "2024-07-23", "2025-02-01", "2026-02-01"]


# ---------------- Phase 2: EDA ----------------
def returns(close, cols):
    return close[cols].pct_change().dropna(how="all")


def return_stats(r):
    r = r.dropna()
    jb = stats.jarque_bera(r)
    return {
        "Mean daily return (%)": r.mean() * 100,
        "Daily volatility (%)": r.std() * 100,
        "Annualised return (%)": r.mean() * 252 * 100,
        "Annualised volatility (%)": r.std() * np.sqrt(252) * 100,
        "Skewness": stats.skew(r),
        "Excess kurtosis": stats.kurtosis(r),
        "Jarque-Bera p-value": jb.pvalue,
        "Worst day (%)": r.min() * 100,
        "Best day (%)": r.max() * 100,
    }


def sector_returns(close, stocks):
    r = returns(close, stocks)
    r.columns = [STOCKS[c] for c in r.columns]
    return r.T.groupby(level=0).mean().T  # equal-weight sector daily returns


def yearly_sector_table(sec_r):
    yr = (1 + sec_r).groupby(sec_r.index.year).prod() - 1
    return (yr * 100).T


def calendar_effects(nifty_r):
    wd = nifty_r.groupby(nifty_r.index.day_name()).mean() * 100
    wd = wd.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]).dropna()
    m = nifty_r.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    mon = m.groupby(m.index.month_name()).mean() * 100
    order = pd.date_range("2000-01-01", periods=12, freq="MS").month_name()
    mon = mon.reindex(order).dropna()
    bd = [pd.Timestamp(d) for d in BUDGET_DAYS if pd.Timestamp(d) in nifty_r.index]
    budget = nifty_r.loc[bd] * 100 if bd else pd.Series(dtype=float)
    return wd, mon, budget


def lead_lag(x, y, lags=10):
    """Correlation of y(t) with x(t-k): positive k means x leads y."""
    df = pd.concat([x, y], axis=1).dropna()
    df.columns = ["x", "y"]
    return pd.Series({k: df["y"].corr(df["x"].shift(k)) for k in range(-lags, lags + 1)})


# ---------------- Phase 3: statistics ----------------
def adf(series):
    from statsmodels.tsa.stattools import adfuller
    s = series.dropna()
    res = adfuller(s, autolag="AIC")
    return {"ADF statistic": res[0], "p-value": res[1],
            "Result": "Stationary" if res[1] < 0.05 else "Non-stationary"}


def granger(cause, effect, maxlag=5):
    from statsmodels.tsa.stattools import grangercausalitytests
    df = pd.concat([effect, cause], axis=1).dropna()
    res = grangercausalitytests(df.values, maxlag=maxlag)
    ps = {lag: res[lag][0]["ssr_ftest"][1] for lag in res}
    best = min(ps, key=ps.get)
    return best, ps[best]


def anova_months(nifty_r):
    groups = [g.values for _, g in nifty_r.groupby(nifty_r.index.month)]
    f, p = stats.f_oneway(*groups)
    return f, p


def anova_weekdays(nifty_r):
    groups = [g.values for _, g in nifty_r.groupby(nifty_r.index.dayofweek)]
    f, p = stats.f_oneway(*groups)
    return f, p


def volume_spike_test(close, volume, stocks, z_thr=2.0):
    """Do volume spikes (a proxy for delivery-backed buying) lead to different next-day returns?"""
    r = close[stocks].pct_change()
    lv = np.log(volume[stocks].replace(0, np.nan))
    z = (lv - lv.rolling(60).mean()) / lv.rolling(60).std()
    nxt = r.shift(-1)
    spike_up = (z > z_thr) & (r > 0)
    a = nxt[spike_up].stack().dropna()
    b = nxt[~spike_up & z.notna()].stack().dropna()
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return {"Spike days": len(a), "Avg next-day return after spike (%)": a.mean() * 100,
            "Avg next-day return otherwise (%)": b.mean() * 100, "t": t, "p": p}


def rolling_avg_corr(r, window=63, step=5):
    out = {}
    for i in range(window, len(r), step):
        c = r.iloc[i - window:i].corr().values
        out[r.index[i]] = c[np.triu_indices_from(c, 1)].mean()
    return pd.Series(out)


def garch(nifty_r):
    from arch import arch_model
    res = arch_model(nifty_r.dropna() * 100, vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
    ann_vol = res.conditional_volatility * np.sqrt(252)
    return res.params, ann_vol


def decompose_monthly(nifty):
    from statsmodels.tsa.seasonal import seasonal_decompose
    m = np.log(nifty.resample("ME").last().dropna())
    return seasonal_decompose(m, model="additive", period=12)


# ---------------- Phase 4: regimes ----------------
def regime_features(close):
    r = np.log(close[INDEX]).diff()
    return pd.DataFrame({
        "trend": r.rolling(21).mean() * 252,
        "vol": r.rolling(21).std() * np.sqrt(252),
        "vix": close[VIX],
    }).dropna()


class RegimeModel:
    """4-state Gaussian HMM (falls back to a Gaussian mixture if hmmlearn is missing)."""

    def __init__(self, seed=42):
        self.seed = seed

    def fit(self, X):
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        try:
            from hmmlearn.hmm import GaussianHMM
            self.model = GaussianHMM(n_components=4, covariance_type="full",
                                     n_iter=300, random_state=self.seed).fit(Z)
            self.kind = "Hidden Markov Model"
        except ImportError:
            from sklearn.mixture import GaussianMixture
            self.model = GaussianMixture(4, random_state=self.seed).fit(Z)
            self.kind = "Gaussian Mixture (hmmlearn not installed)"
        raw = self.model.predict(Z)
        prof = X.groupby(raw).mean()
        panic = prof["vol"].idxmax()
        rest = prof.drop(panic).sort_values("trend").index.tolist()  # bear, sideways, bull
        self.map = {panic: "Panic", rest[0]: "Bear", rest[1]: "Sideways", rest[2]: "Bull"}
        return self

    def label(self, X):
        """Labels using the whole sequence (for historical analysis)."""
        return pd.Series(self.model.predict(self.scaler.transform(X)), index=X.index).map(self.map)

    def label_filtered(self, X, start):
        """Labels using only data up to each day (no look-ahead), for days >= start."""
        Z = self.scaler.transform(X)
        out = {}
        hmm = hasattr(self.model, "transmat_")
        for i in np.where(X.index >= start)[0]:
            st = self.model.predict_proba(Z[: i + 1])[-1].argmax() if hmm else self.model.predict(Z[i:i + 1])[0]
            out[X.index[i]] = self.map[st]
        return pd.Series(out)


def regime_profile(labels, nifty_r):
    df = pd.DataFrame({"reg": labels, "r": nifty_r}).dropna()
    runs = (df["reg"] != df["reg"].shift()).cumsum()
    dur = df.groupby(runs)["reg"].agg(["first", "size"])
    rows = []
    for g in REGIMES:
        d = df[df["reg"] == g]["r"]
        if d.empty:
            continue
        rows.append({"Regime": g, "Days": len(d), "Share of days (%)": len(d) / len(df) * 100,
                     "Ann. return (%)": d.mean() * 252 * 100,
                     "Ann. volatility (%)": d.std() * np.sqrt(252) * 100,
                     "Episodes": int((dur["first"] == g).sum()),
                     "Avg duration (days)": dur[dur["first"] == g]["size"].mean()})
    return pd.DataFrame(rows).set_index("Regime")


def transition_matrix(labels):
    s = labels.dropna()
    t = pd.crosstab(s.shift().iloc[1:], s.iloc[1:], normalize="index")
    order = [g for g in REGIMES if g in t.index]
    return t.reindex(index=order, columns=order).fillna(0) * 100


def corr_by_regime(r, labels):
    out = {}
    for g in REGIMES:
        d = r.loc[r.index.intersection(labels[labels == g].index)]
        if len(d) > 30:
            c = d.corr().values
            out[g] = c[np.triu_indices_from(c, 1)].mean()
    return pd.Series(out)


def sector_by_regime(sec_r, labels):
    df = sec_r.join(labels.rename("reg"), how="inner")
    return (df.groupby("reg").mean() * 252 * 100).reindex([g for g in REGIMES if g in df["reg"].unique()]).T


# ---------------- Phase 4: clustering and PCA ----------------
def stock_features(close, stocks, labels=None):
    r = returns(close, stocks)
    m = close[INDEX].pct_change().reindex(r.index)
    dv = close[VIX].pct_change().reindex(r.index)
    rows = {}
    for s in stocks:
        x = r[s].dropna()
        mm = m.reindex(x.index)
        down = mm < 0
        p = close[s].dropna()
        rows[short(s)] = {
            "Ann. return (%)": x.mean() * 252 * 100,
            "Ann. volatility (%)": x.std() * np.sqrt(252) * 100,
            "Beta": x.cov(mm) / mm.var(),
            "Downside beta": x[down].cov(mm[down]) / mm[down].var(),
            "Max drawdown (%)": ((p / p.cummax()) - 1).min() * 100,
            "Corr with VIX change": x.corr(dv.reindex(x.index)),
        }
    return pd.DataFrame(rows).T


def cluster_stocks(feat, k):
    Z = StandardScaler().fit_transform(feat)
    sil = {kk: silhouette_score(Z, KMeans(kk, n_init=10, random_state=0).fit_predict(Z))
           for kk in range(2, min(8, len(feat)))}
    km = KMeans(k, n_init=10, random_state=0).fit(Z)
    lab = pd.Series(km.labels_, index=feat.index)
    # name clusters by their character
    prof = feat.groupby(lab).mean()
    names = {}
    for c, row in prof.iterrows():
        if row["Beta"] == prof["Beta"].max():
            names[c] = "High-beta / aggressive"
        elif row["Beta"] == prof["Beta"].min():
            names[c] = "Defensive / low-beta"
        elif row["Ann. return (%)"] == prof["Ann. return (%)"].max():
            names[c] = "Steady compounders"
        else:
            names[c] = f"Market-like group {c + 1}"
    return lab.map(names), pd.Series(sil), Z


def pca_factors(close, stocks, n=5):
    r = returns(close, stocks).dropna()
    Z = StandardScaler().fit_transform(r)
    p = PCA(n_components=min(n, len(stocks))).fit(Z)
    load = pd.DataFrame(p.components_.T, index=[short(s) for s in stocks],
                        columns=[f"PC{i + 1}" for i in range(p.n_components_)])
    if load["PC1"].mean() < 0:
        load["PC1"] *= -1
    return pd.Series(p.explained_variance_ratio_ * 100, index=load.columns), load
