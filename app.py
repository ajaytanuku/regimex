"""RegimeX - Analytics-Driven Study of Indian Stock Market Regimes (Streamlit web app)."""
import io
from datetime import date

import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st

from src import analytics as A
from src import data as D
from src import live as L
from src import predict as P

st.set_page_config(page_title="RegimeX", page_icon=":material/monitoring:", layout="wide")

# ---------------- colours (validated palette, light and dark steps) ----------------
DARK = getattr(getattr(st.context, "theme", None), "type", "light") == "dark"
SERIES = (["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"] if DARK else
          ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"])
REG_C = ({"Bull": "#199e70", "Sideways": "#3987e5", "Bear": "#c98500", "Panic": "#e66767"} if DARK else
         {"Bull": "#1baf7a", "Sideways": "#2a78d6", "Bear": "#eda100", "Panic": "#e34948"})
MUTED = "#898781"
MID = "#383835" if DARK else "#f0efec"
DIVERGING = [[0, SERIES[7]], [0.5, MID], [1, SERIES[0]]]          # red = negative, blue = positive
SEQUENTIAL = [[0, "#0d366b"], [1, "#9ec5f4"]] if DARK else [[0, "#cde2fb"], [1, "#2a78d6"]]


def style(fig, h=380, title=None, legend=True):
    top = (40 if title else 10) + (34 if legend else 0)
    fig.update_layout(height=h, showlegend=legend, margin=dict(l=10, r=10, t=top, b=10),
                      title=dict(text=title, x=0, xanchor="left", y=1, yref="container", yanchor="top", pad=dict(t=8)),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch")


def table(df, fmt="{:.2f}"):
    st.dataframe(df.style.format(fmt, na_rep="-"), width="stretch")


def pfmt(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


# ---------------- sidebar ----------------
with st.sidebar:
    st.title("RegimeX")
    st.caption("Indian stock market regime analytics")
    page = st.radio("Go to", ["Home", "Live market", "Paper trading", "1. Data & quality",
                              "2. Exploratory analysis", "3. Statistical tests", "4. Market regimes",
                              "5. Clustering & PCA", "6. Prediction & portfolio", "7. Key insights"])
    st.divider()
    source = st.radio("Data source", ["Live NSE data (Yahoo Finance)", "Demo data (offline)"])
    refresh = st.select_slider("Live prices refresh every (seconds)", [10, 15, 30, 60], value=15)
    if st.button("Reload daily data", help="Download the latest daily history again"):
        st.cache_data.clear()
    start_year = st.slider("Start year", 2012, 2022, 2015)
    names = {D.short(t): t for t in D.STOCKS}
    picked = st.multiselect("Stocks (Nifty 50)", list(names), default=list(names))
    test_start = st.date_input("Backtest starts on", date(2024, 1, 1),
                               min_value=date(start_year + 4, 1, 1), max_value=date.today())
    fii_file = st.file_uploader("Optional: FII/DII flows CSV", type="csv",
                                help="Columns: Date, FII_Net, DII_Net (Rs crore). Download from the NSE website.")

if len(picked) < 5:
    st.warning("Pick at least 5 stocks in the sidebar.")
    st.stop()

TICKERS = tuple(names[n] for n in picked)
START = f"{start_year}-01-01"
LIVE = source.startswith("Live")


# ---------------- cached pipeline ----------------
@st.cache_data(show_spinner="Downloading and cleaning market data...", ttl=6 * 3600)
def get_data(live, start, tickers):
    used, err = "Demo data", ""
    if live:
        try:
            raw_close, raw_vol = D.load_live(tickers, start)
            used = "Live NSE data (Yahoo Finance)"
        except Exception as e:  # network or Yahoo issue -> fall back
            err = str(e)
            raw_close, raw_vol = D.load_demo(tickers, start)
    else:
        raw_close, raw_vol = D.load_demo(tickers, start)
    close, vol, stocks, report = D.clean(raw_close.copy(), raw_vol.copy(), tickers)
    return raw_close, close, vol, stocks, report, used, err


@st.cache_data(show_spinner="Detecting market regimes...")
def get_regimes(live, start, tickers):
    _, close, *_ = get_data(live, start, tickers)
    X = A.regime_features(close)
    rm = A.RegimeModel().fit(X)
    return rm.label(X), rm.kind


@st.cache_resource(show_spinner="Training regime-aware models...")
def get_models(live, start, tickers, test_start):
    _, close, vol, stocks, *_ = get_data(live, start, tickers)
    return P.run_models(close, vol, stocks, test_start)


@st.cache_data(show_spinner="Running statistical tests...")
def get_stats(live, start, tickers, fii_bytes):
    _, close, vol, stocks, *_ = get_data(live, start, tickers)
    nr = close[D.INDEX].pct_change().dropna()
    vix_chg = close[D.VIX].pct_change()
    rows = []

    def add(h, test, stat, p, yes, no):
        rows.append({"Hypothesis": h, "Test": test, "Statistic": stat, "p-value": p,
                     "Result (5% level)": yes if p < 0.05 else no})

    jb = A.return_stats(nr)
    add("Nifty daily returns are normally distributed", "Jarque-Bera", np.nan, jb["Jarque-Bera p-value"],
        "Rejected: returns have fat tails", "Not rejected")
    lag, p = A.granger(vix_chg, nr)
    add("India VIX changes help predict Nifty returns", f"Granger causality (best lag {lag})", np.nan, p,
        "Supported", "No evidence")
    lag, p = A.granger(nr, vix_chg)
    add("Nifty returns help predict India VIX changes", f"Granger causality (best lag {lag})", np.nan, p,
        "Supported", "No evidence")
    vs = A.volume_spike_test(close, vol, stocks)
    add("Up-days with volume spikes lead to different next-day returns", "Welch t-test", vs["t"], vs["p"],
        "Supported", "No evidence")
    f, p = A.anova_months(nr)
    add("Average Nifty return differs by calendar month", "One-way ANOVA", f, p, "Supported", "No evidence")
    f, p = A.anova_weekdays(nr)
    add("Average Nifty return differs by weekday", "One-way ANOVA", f, p, "Supported", "No evidence")
    fii = None
    if fii_bytes:
        fii = D.fii_csv(io.BytesIO(fii_bytes))
        if "FII_Net" in fii:
            lag, p = A.granger(fii["FII_Net"], nr)
            add("FII net flows help predict Nifty returns", f"Granger causality (best lag {lag})", np.nan, p,
                "Supported", "No evidence")
    tests = pd.DataFrame(rows)
    adf = pd.DataFrame({"Nifty price level": A.adf(close[D.INDEX]),
                        "Nifty daily returns": A.adf(nr)}).T
    params, gvol = A.garch(nr)
    return tests, adf, params, gvol, vs, fii


@st.cache_data(ttl=10, show_spinner=False)
def get_feed(live, start, tickers):
    """Live quotes when real data is available, otherwise the simulated demo feed."""
    _, close, _, _, _, used, _ = get_data(live, start, tickers)
    err = ""
    if used.startswith("Live"):
        try:
            q, intra, is_open = L.live_feed(tickers)
            return q.dropna(subset=["Price"]), intra, is_open, "Live NSE prices (Yahoo Finance)", ""
        except Exception as e:
            err = str(e)
    q, intra, is_open = L.demo_feed(tickers, close.iloc[-1])
    return q.dropna(subset=["Price"]), intra, is_open, "Simulated demo feed", err


try:
    raw_close, close, vol, stocks, report, used, live_err = get_data(LIVE, START, TICKERS)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

if LIVE and used != source:
    st.warning("Live download failed, so the app is showing demo data. Check your internet connection "
               f"or try again later. ({live_err or 'unknown error'})")

nifty = close[D.INDEX]
nr = nifty.pct_change().dropna()
rets = A.returns(close, stocks)
labels, regime_kind = get_regimes(LIVE, START, TICKERS)
fii_bytes = fii_file.getvalue() if fii_file else None


def regime_chart(h=420, events=True):
    fig = go.Figure(go.Scatter(x=nifty.index, y=nifty, mode="lines", line=dict(color=MUTED, width=1),
                               name="Nifty 50", hoverinfo="skip", showlegend=False))
    for g in A.REGIMES:
        idx = labels.index[labels == g]
        fig.add_trace(go.Scatter(x=idx, y=nifty.reindex(idx), mode="markers", name=g,
                                 marker=dict(color=REG_C[g], size=4),
                                 hovertemplate="%{x|%d %b %Y}<br>Nifty %{y:,.0f}<br>" + g + "<extra></extra>"))
    if events:
        for d, txt in A.EVENTS.items():
            d = pd.Timestamp(d)
            if nifty.index[0] <= d <= nifty.index[-1]:
                fig.add_shape(type="line", x0=d, x1=d, y0=0, y1=1, yref="paper",
                              line=dict(color=MUTED, width=1, dash="dot"))
                fig.add_annotation(x=d, y=0.99, yref="paper", text=txt, showarrow=False, xanchor="left",
                                   yanchor="top", xshift=4, font=dict(size=11, color=MUTED))
    fig.update_yaxes(title="Nifty 50")
    return style(fig, h)


# ================================ PAGES ================================
if page == "Home":
    st.title("RegimeX: Analytics-Driven Study of Indian Stock Market Regimes")
    st.write("How does the Indian market behave in bull, bear, sideways and panic phases, which factors "
             "drive stocks in each phase, and can those insights build a better portfolio?")
    c = st.columns(5)
    c[0].metric("Period", f"{close.index[0]:%Y}–{close.index[-1]:%Y}")
    c[1].metric("Stocks analysed", len(stocks))
    c[2].metric("Current regime", labels.iloc[-1])
    yr = nifty.iloc[-1] / nifty.iloc[-253] - 1 if len(nifty) > 253 else np.nan
    c[3].metric("Nifty 50", f"{nifty.iloc[-1]:,.0f}", f"{yr * 100:+.1f}% in 1 year")
    c[4].metric("India VIX", f"{close[D.VIX].iloc[-1]:.1f}")
    st.caption(f"Data source: {used}")
    st.subheader("Nifty 50 coloured by detected market regime")
    show(regime_chart())
    st.subheader("How the project works")
    st.markdown("""
| Phase | Page | Analytics type |
|---|---|---|
| Data collection and cleaning | 1. Data & quality | Descriptive |
| Returns, volatility, sectors, calendar effects | 2. Exploratory analysis | Descriptive |
| Correlation, hypothesis tests, GARCH | 3. Statistical tests | Diagnostic |
| Hidden Markov Model regimes, clustering, PCA | 4 and 5 | Diagnostic |
| Regime-aware forecasting and backtest | 6. Prediction & portfolio | Predictive and prescriptive |
| Findings in plain words | 7. Key insights | Summary |
| Live prices and trading with virtual money | Live market, Paper trading | Real-time application |
""")
    st.caption("Educational project. Not investment advice.")

elif page == "Live market":
    st.title("Live market")

    @st.fragment(run_every=refresh)
    def live_view():
        q, intra, is_open, src, err = get_feed(LIVE, START, TICKERS)
        if err:
            st.warning(f"Live quotes failed, so the simulated feed is shown. ({err})")
        stk = q.drop([D.INDEX, D.VIX], errors="ignore")
        n, v = q.loc[D.INDEX], q.loc[D.VIX]
        c = st.columns(5)
        c[0].metric("Nifty 50", f"{n['Price']:,.2f}", f"{n['Change']:+,.2f} ({n['Change %']:+.2f}%)")
        c[1].metric("India VIX", f"{v['Price']:.2f}", f"{v['Change %']:+.2f}%", delta_color="inverse")
        c[2].metric("Advances / declines", f"{(stk['Change'] > 0).sum()} / {(stk['Change'] < 0).sum()}")
        c[3].metric("Model regime", labels.iloc[-1])
        c[4].metric("Market", "Simulated" if src.startswith("Simulated") else ("Open" if is_open else "Closed"))
        st.caption(f"{src} · updated {L.now_ist():%d %b %Y, %H:%M:%S} IST · refreshes every {refresh} s. "
                   "NSE trades 9:15 am to 3:30 pm IST, Monday to Friday"
                   + ("." if is_open else "; outside those hours prices stay at the last close."))

        if len(intra):
            fig = go.Figure(go.Scatter(x=intra.index, y=intra.values, mode="lines", name="Nifty 50",
                                       line=dict(color=SERIES[0], width=2),
                                       hovertemplate="%{x|%H:%M}<br>%{y:,.2f}<extra></extra>"))
            fig.add_hline(y=n["Prev close"], line=dict(color=MUTED, width=1, dash="dot"),
                          annotation_text="Previous close", annotation_position="bottom right",
                          annotation_font_color=MUTED)
            fig.update_xaxes(tickformat="%H:%M")
            show(style(fig, 360, f"Nifty 50 intraday, 1-minute ({intra.index[-1]:%d %b %Y})", legend=False))

        c1, c2 = st.columns([2, 3])
        s = stk["Change %"].sort_values()
        with c1:
            fig = go.Figure(go.Bar(x=s.values, y=[D.short(t) for t in s.index], orientation="h",
                                   marker_color=[SERIES[7] if x < 0 else SERIES[0] for x in s.values],
                                   hovertemplate="%{y}: %{x:+.2f}%<extra></extra>"))
            fig.update_xaxes(title="Change today (%)")
            show(style(fig, max(420, 18 * len(s)), "Today's movers", legend=False))
        with c2:
            wl = pd.DataFrame({"Stock": [D.short(t) for t in stk.index],
                               "Sector": [D.STOCKS.get(t, "") for t in stk.index],
                               "Price (Rs)": stk["Price"].values, "Change (Rs)": stk["Change"].values,
                               "Change %": stk["Change %"].values}).sort_values("Change %", ascending=False)
            st.dataframe(wl.style.format({"Price (Rs)": "{:,.2f}", "Change (Rs)": "{:+,.2f}", "Change %": "{:+.2f}"}),
                         hide_index=True, width="stretch", height=max(420, 18 * len(s)))

    live_view()

elif page == "Paper trading":
    st.title("Paper trading")
    st.caption(f"Trade Nifty 50 stocks with Rs {L.START_CASH:,.0f} of virtual money at live prices. Charges of "
               f"{L.CHARGES:.1%} per order are deducted. No real money is used.")
    user = st.text_input("Trader name", "guest", help="Trades are saved under this name. Use the same name next time.")
    book = L.Book(user)
    q, intra, is_open, src, err = get_feed(LIVE, START, TICKERS)
    prices = q["Price"]
    tradable = [s for s in picked if names[s] in prices.index]
    if not is_open:
        st.info("The market is closed, so orders fill at the last available price.")

    c1, c2 = st.columns(2, border=True)
    with c1:
        st.subheader("Place an order")
        with st.form("order", border=False):
            stock = st.selectbox("Stock", tradable)
            a, b = st.columns(2)
            side = a.radio("Side", ["BUY", "SELL"], horizontal=True)
            qty = b.number_input("Quantity", min_value=1, value=10, step=1)
            placed = st.form_submit_button("Place market order", type="primary")
        if placed:
            px = prices[names[stock]]
            try:
                book.order(names[stock], side, qty, px)
                st.success(f"{side} {qty} x {stock} at Rs {px:,.2f}")
            except ValueError as e:
                st.error(str(e))
    with c2:
        st.subheader("Auto-trade with RegimeX")
        st.write("Rebalance the whole account to the model's current recommended portfolio. It sells what the "
                 "model no longer holds, then buys the new weights. In Bear or Panic regimes it keeps more cash.")
        top_n = st.slider("Stocks to hold", 4, 12, 8)
        if st.button("Rebalance to RegimeX portfolio"):
            res = get_models(LIVE, START, TICKERS, test_start)
            _, reg, port = P.current_portfolio(close, vol, stocks, res, top_n)
            target = {names[s]: w / 100 for s, w in zip(port["Stock"], port["Weight (%)"])
                      if w > 0.5 and names[s] in prices.index}
            try:
                done = L.rebalance(book, target, prices)
                st.success(f"Regime: {reg}. Placed {len(done)} orders: "
                           + ", ".join(f"{side} {qty} {D.short(t)}" for side, t, qty in done))
            except ValueError as e:
                st.error(str(e))

    @st.fragment(run_every=refresh)
    def account():
        q, *_ = get_feed(LIVE, START, TICKERS)
        hold, cash = book.holdings()
        ltp = hold["stock"].map(q["Price"])
        h = pd.DataFrame({"Stock": hold["stock"].map(D.short), "Qty": hold["qty"], "Avg price": hold["avg_price"],
                          "LTP": ltp, "Value": ltp * hold["qty"], "P&L": (ltp - hold["avg_price"]) * hold["qty"],
                          "P&L %": (ltp / hold["avg_price"] - 1) * 100})
        equity = cash + h["Value"].sum()
        pnl = equity - L.START_CASH
        st.subheader("Your account")
        c = st.columns(4)
        c[0].metric("Account value", f"Rs {equity:,.0f}", f"{pnl:+,.0f} ({pnl / L.START_CASH * 100:+.2f}%)")
        c[1].metric("Cash", f"Rs {cash:,.0f}")
        c[2].metric("Holdings value", f"Rs {h['Value'].sum():,.0f}")
        c[3].metric("Unrealised P&L", f"Rs {h['P&L'].sum():+,.0f}")
        st.caption(f"Prices: {src} · updated {L.now_ist():%H:%M:%S} IST · LTP = last traded price")
        if h.empty:
            st.info("No holdings yet. Place an order or use Auto-trade above.")
        else:
            st.dataframe(h.sort_values("Value", ascending=False).style.format(
                {"Avg price": "{:,.2f}", "LTP": "{:,.2f}", "Value": "{:,.0f}", "P&L": "{:+,.0f}", "P&L %": "{:+.2f}"}),
                hide_index=True, width="stretch")

    account()

    tr = book.trades()
    with st.expander(f"Trade history ({len(tr)} orders)"):
        if len(tr):
            t = tr.iloc[::-1].assign(stock=tr["stock"].map(D.short), value=tr["qty"] * tr["price"])
            t.columns = ["Time (IST)", "Stock", "Side", "Qty", "Price", "Charges", "Type", "Value"]
            st.dataframe(t.style.format({"Price": "{:,.2f}", "Charges": "{:,.2f}", "Value": "{:,.0f}"}),
                         hide_index=True, width="stretch")
            st.download_button("Download trades (CSV)", t.to_csv(index=False).encode(), "regimex_trades.csv")
        else:
            st.write("No trades yet.")
    with st.expander("Reset account"):
        st.write("Deletes all trades for this trader name and restores the starting cash.")
        if st.button("Reset account"):
            book.reset()
            st.rerun()
    st.caption("Trades are stored on the server under your trader name. On the free online version they reset "
               "if the app restarts.")

elif page == "1. Data & quality":
    st.title("Phase 1: Data collection and cleaning")
    c = st.columns(4)
    c[0].metric("Trading days", f"{len(close):,}")
    c[1].metric("Stocks kept", (report["Status"] == "Kept").sum())
    c[2].metric("Stocks dropped", (report["Status"] == "Dropped").sum())
    c[3].metric("Source", "Live" if used.startswith("Live") else "Demo")
    st.write("Cleaning steps: keep only NSE trading days, drop stocks with more than 20% missing history, "
             "forward-fill gaps of up to 5 days, adjust prices for splits and bonuses (adjusted close), "
             "and flag extreme return days (|z| > 6) for review.")

    st.subheader("Missing values in the raw data (% of days per month)")
    rc = raw_close[[t for t in TICKERS if t in raw_close]]
    rc = rc[raw_close[D.INDEX].notna()]
    miss = rc.isna().resample("ME").mean().T * 100
    miss.index = [D.short(t) for t in miss.index]
    fig = go.Figure(go.Heatmap(z=miss.values, x=miss.columns, y=miss.index, colorscale=SEQUENTIAL,
                               zmin=0, colorbar=dict(title="% missing"),
                               hovertemplate="%{y}<br>%{x|%b %Y}<br>%{z:.1f}% missing<extra></extra>"))
    show(style(fig, max(380, 16 * len(miss)), legend=False))

    st.subheader("Data quality report")
    table(report, {"Missing % (raw)": "{:.2f}", "Missing % (clean)": "{:.2f}",
                   "Extreme return days (|z|>6)": "{:.0f}"})
    with st.expander("Preview cleaned prices"):
        prev = close[stocks].tail(10).copy()
        prev.columns = [D.short(c) for c in prev.columns]
        table(prev)
    st.download_button("Download cleaned prices (CSV)", close.to_csv().encode(), "regimex_clean_prices.csv")

elif page == "2. Exploratory analysis":
    st.title("Phase 2: Exploratory data analysis")
    t1, t2, t3, t4, t5 = st.tabs(["Returns", "Volatility", "Sectors", "Calendar effects", "Money flows"])

    with t1:
        rs = A.return_stats(nr)
        c = st.columns(4)
        c[0].metric("Annualised return", f"{rs['Annualised return (%)']:.1f}%")
        c[1].metric("Annualised volatility", f"{rs['Annualised volatility (%)']:.1f}%")
        c[2].metric("Skewness", f"{rs['Skewness']:.2f}")
        c[3].metric("Excess kurtosis", f"{rs['Excess kurtosis']:.1f}")
        x = nr * 100
        grid = np.linspace(x.min(), x.max(), 200)
        bw = (x.max() - x.min()) / 80
        normal = len(x) * bw * np.exp(-0.5 * ((grid - x.mean()) / x.std()) ** 2) / (x.std() * np.sqrt(2 * np.pi))
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=x, xbins=dict(size=bw), name="Actual returns", marker_color=SERIES[0]))
        fig.add_trace(go.Scatter(x=grid, y=normal, name="Normal distribution", line=dict(color=SERIES[1], width=2)))
        fig.update_xaxes(title="Nifty daily return (%)")
        fig.update_yaxes(title="Days")
        show(style(fig, 400, "Nifty daily returns vs a normal distribution"))
        st.caption(f"Excess kurtosis of {rs['Excess kurtosis']:.1f} means extreme days happen far more often "
                   f"than a normal distribution predicts (worst day {rs['Worst day (%)']:.1f}%, "
                   f"best day {rs['Best day (%)']:.1f}%).")

    with t2:
        fig = go.Figure(go.Scatter(x=nifty.index, y=nifty, line=dict(color=SERIES[0], width=2), name="Nifty 50"))
        fig.update_yaxes(title="Index level")
        show(style(fig, 300, "Nifty 50", legend=False))
        rv = nr.rolling(21).std() * np.sqrt(252) * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rv.index, y=rv, name="Realised volatility (21-day)", line=dict(color=SERIES[0], width=2)))
        fig.add_trace(go.Scatter(x=close.index, y=close[D.VIX], name="India VIX", line=dict(color=SERIES[1], width=2)))
        fig.update_yaxes(title="Annualised volatility (%)")
        show(style(fig, 340, "Realised volatility and India VIX"))
        from statsmodels.tsa.stattools import acf
        lags = np.arange(1, 21)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=lags, y=acf(nr, nlags=20)[1:], name="Returns", marker_color=SERIES[0]))
        fig.add_trace(go.Bar(x=lags, y=acf(nr.abs(), nlags=20)[1:], name="Absolute returns", marker_color=SERIES[1]))
        fig.update_xaxes(title="Lag (days)")
        fig.update_yaxes(title="Autocorrelation")
        show(style(fig, 340, "Volatility clustering: big moves follow big moves"))
        dv = close[D.VIX].pct_change().reindex(nr.index) * 100
        fig = go.Figure(go.Scatter(x=dv, y=nr * 100, mode="markers", marker=dict(color=SERIES[0], size=5, opacity=0.5),
                                   hovertemplate="VIX change %{x:.1f}%<br>Nifty %{y:.2f}%<extra></extra>"))
        fig.update_xaxes(title="India VIX daily change (%)")
        fig.update_yaxes(title="Nifty daily return (%)")
        show(style(fig, 380, f"Nifty return vs VIX change (correlation {dv.corr(nr * 100):.2f})", legend=False))

    with t3:
        sec = A.sector_returns(close, stocks)
        yt = A.yearly_sector_table(sec)
        fig = go.Figure(go.Heatmap(z=yt.values, x=[str(c) for c in yt.columns], y=yt.index, colorscale=DIVERGING,
                                   zmid=0, texttemplate="%{z:.0f}", colorbar=dict(title="Return %"),
                                   hovertemplate="%{y} %{x}<br>%{z:.1f}%<extra></extra>"))
        show(style(fig, 60 + 32 * len(yt), "Sector rotation: yearly return by sector (%)", legend=False))
        cum = (1 + sec).cumprod()
        fig = go.Figure()
        order = cum.iloc[-1].sort_values(ascending=False).index
        for i, s in enumerate(order):
            fig.add_trace(go.Scatter(x=cum.index, y=cum[s], name=s, line=dict(width=2, color=SERIES[i % 8]),
                                     visible=True if i < 4 else "legendonly"))
        fig.update_yaxes(title="Growth of Rs 1")
        show(style(fig, 400, "Cumulative sector performance (top 4 shown, click legend for more)"))

    with t4:
        wd, mon, budget = A.calendar_effects(nr)
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure(go.Bar(x=wd.index, y=wd.values, marker_color=SERIES[0],
                                   hovertemplate="%{x}: %{y:.3f}%<extra></extra>"))
            fig.update_yaxes(title="Avg daily return (%)")
            show(style(fig, 340, "Day-of-week effect", legend=False))
        with c2:
            fig = go.Figure(go.Bar(x=mon.index.str[:3], y=mon.values, marker_color=SERIES[0],
                                   hovertemplate="%{x}: %{y:.2f}%<extra></extra>"))
            fig.update_yaxes(title="Avg monthly return (%)")
            show(style(fig, 340, "Month-of-year effect", legend=False))
        st.subheader("Union Budget days")
        if len(budget):
            c = st.columns(3)
            c[0].metric("Budget days in data", len(budget))
            c[1].metric("Avg Nifty move on Budget day", f"{budget.mean():+.2f}%")
            c[2].metric("Budget-day move vs a normal day", f"{budget.abs().mean() / (nr.abs().mean() * 100):.1f}x",
                        help="Average absolute Nifty move on Budget days divided by the average absolute move on all days")
            bt = budget.rename("Nifty return (%)").to_frame()
            bt.index = bt.index.strftime("%d %b %Y")
            table(bt)
        else:
            st.info("No Budget days fall inside the selected period.")

    with t5:
        if fii_bytes:
            fii = D.fii_csv(io.BytesIO(fii_bytes))
            col = "FII_Net"
            fig = go.Figure()
            for i, c_ in enumerate(fii.columns):
                fig.add_trace(go.Bar(x=fii.index, y=fii[c_], name=c_.replace("_", " "), marker_color=SERIES[i]))
            fig.update_yaxes(title="Net flow (Rs crore)")
            show(style(fig, 340, "FII and DII net flows"))
            if col in fii:
                ll = A.lead_lag(fii[col], nr)
                fig = go.Figure(go.Bar(x=ll.index, y=ll.values, marker_color=SERIES[0],
                                       hovertemplate="Lag %{x}: corr %{y:.3f}<extra></extra>"))
                fig.update_xaxes(title="Lag in days (positive = FII flow comes first)")
                fig.update_yaxes(title="Correlation with Nifty return")
                show(style(fig, 340, "Lead-lag: do FII flows move before Nifty?", legend=False))
        else:
            st.info("Upload an FII/DII flows CSV in the sidebar to analyse money flows. "
                    "Format: `Date, FII_Net, DII_Net` with net values in Rs crore "
                    "(download from NSE: Market data > FII/DII trading activity).")

elif page == "3. Statistical tests":
    st.title("Phase 3: Statistical analysis")
    tests, adf, params, gvol, vs, fii = get_stats(LIVE, START, TICKERS, fii_bytes)

    st.subheader("Hypothesis tests")
    st.dataframe(tests.style.format({"Statistic": "{:.2f}", "p-value": pfmt}, na_rep="-"),
                 width="stretch", hide_index=True)
    st.caption(f"Volume-spike test: after up-days with a volume spike (log volume z-score > 2), the next-day "
               f"return averaged {vs['Avg next-day return after spike (%)']:.3f}% vs "
               f"{vs['Avg next-day return otherwise (%)']:.3f}% on other days ({vs['Spike days']:,} spike days). "
               "Volume is used as a proxy for delivery-backed buying.")

    st.subheader("Correlation between stocks")
    cm = rets.corr()
    cm.index = cm.columns = [D.short(c) for c in cm.columns]
    fig = go.Figure(go.Heatmap(z=cm.values, x=cm.columns, y=cm.index, colorscale=DIVERGING, zmid=0, zmin=-1, zmax=1,
                               colorbar=dict(title="Correlation"),
                               hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>"))
    show(style(fig, max(450, 18 * len(cm)), legend=False))
    rc = A.rolling_avg_corr(rets)
    fig = go.Figure(go.Scatter(x=rc.index, y=rc, line=dict(color=SERIES[0], width=2), name="Avg correlation"))
    fig.update_yaxes(title="Average pairwise correlation")
    show(style(fig, 320, "Correlations rise during crashes (63-day rolling average)", legend=False))

    st.subheader("Stationarity (Augmented Dickey-Fuller)")
    table(adf, {"ADF statistic": "{:.2f}", "p-value": pfmt})
    st.caption("Prices wander (non-stationary), while returns are stationary, so models are built on returns.")

    st.subheader("Seasonal decomposition of monthly Nifty (log scale)")
    dec = A.decompose_monthly(nifty)
    c = st.columns(3)
    for col_, (name, s) in zip(c, [("Trend", dec.trend), ("Seasonal", dec.seasonal), ("Residual", dec.resid)]):
        with col_:
            fig = go.Figure(go.Scatter(x=s.index, y=s, line=dict(color=SERIES[0], width=2)))
            show(style(fig, 260, name, legend=False))

    st.subheader("GARCH(1,1) volatility model")
    c1, c2 = st.columns([1, 3])
    with c1:
        g = params.rename("Value").to_frame()
        table(g, "{:.4f}")
        persistence = params.get("alpha[1]", 0) + params.get("beta[1]", 0)
        st.metric("Volatility persistence (alpha + beta)", f"{persistence:.3f}")
    with c2:
        fig = go.Figure(go.Scatter(x=gvol.index, y=gvol, line=dict(color=SERIES[0], width=2)))
        fig.update_yaxes(title="Annualised volatility (%)")
        show(style(fig, 340, "GARCH conditional volatility of Nifty", legend=False))

elif page == "4. Market regimes":
    st.title("Phase 4: Market regime detection")
    runs = (labels != labels.shift()).cumsum()
    c = st.columns(3)
    c[0].metric("Current regime", labels.iloc[-1])
    c[1].metric("Days in current regime", int((runs == runs.iloc[-1]).sum()))
    c[2].metric("Model", "HMM (4 states)" if "Hidden" in regime_kind else "Gaussian Mixture")
    show(regime_chart(460))
    with st.expander("How regimes are detected"):
        st.write("A 4-state Gaussian Hidden Markov Model is fitted on three daily features: the 21-day Nifty "
                 "trend (annualised mean return), 21-day realised volatility, and the India VIX level. The state "
                 "with the highest volatility is labelled Panic; the other three are ordered by trend into Bear, "
                 "Sideways and Bull.")

    prof = A.regime_profile(labels, nr)
    st.subheader("Regime profile")
    table(prof, {"Days": "{:,.0f}", "Share of days (%)": "{:.1f}", "Ann. return (%)": "{:.1f}",
                 "Ann. volatility (%)": "{:.1f}", "Episodes": "{:.0f}", "Avg duration (days)": "{:.0f}"})

    c1, c2 = st.columns(2)
    with c1:
        tm = A.transition_matrix(labels)
        fig = go.Figure(go.Heatmap(z=tm.values, x=tm.columns, y=tm.index, colorscale=SEQUENTIAL, zmin=0, zmax=100,
                                   texttemplate="%{z:.1f}%", colorbar=dict(title="%"),
                                   hovertemplate="From %{y} to %{x}: %{z:.1f}%<extra></extra>"))
        fig.update_xaxes(title="Next day")
        fig.update_yaxes(title="Today", autorange="reversed")
        show(style(fig, 380, "Regime transition probabilities", legend=False))
    with c2:
        cr = A.corr_by_regime(rets, labels)
        fig = go.Figure(go.Bar(x=cr.index, y=cr.values, marker_color=[REG_C[g] for g in cr.index],
                               text=[f"{v:.2f}" for v in cr.values], textposition="outside",
                               hovertemplate="%{x}: %{y:.2f}<extra></extra>"))
        fig.update_yaxes(title="Average pairwise correlation", range=[0, 1])
        show(style(fig, 380, "Diversification breaks down in panic", legend=False))

    sr = A.sector_by_regime(A.sector_returns(close, stocks), labels)
    fig = go.Figure(go.Heatmap(z=sr.values, x=sr.columns, y=sr.index, colorscale=DIVERGING, zmid=0,
                               texttemplate="%{z:.0f}", colorbar=dict(title="Ann. return %"),
                               hovertemplate="%{y} in %{x}: %{z:.1f}%<extra></extra>"))
    show(style(fig, 60 + 32 * len(sr), "Annualised sector return in each regime (%)", legend=False))

elif page == "5. Clustering & PCA":
    st.title("Phase 4: Stock clustering and PCA")
    feat = A.stock_features(close, stocks)
    k = st.slider("Number of clusters", 2, 6, 4)
    clusters, sil, _ = A.cluster_stocks(feat, k)
    var, load = A.pca_factors(close, stocks)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(go.Bar(x=sil.index, y=sil.values, marker_color=[SERIES[0] if i != k else SERIES[1] for i in sil.index],
                               hovertemplate="k=%{x}: %{y:.3f}<extra></extra>"))
        fig.update_xaxes(title="Number of clusters", dtick=1)
        fig.update_yaxes(title="Silhouette score")
        show(style(fig, 340, "Choosing k (higher is better)", legend=False))
    with c2:
        fig = go.Figure(go.Bar(x=var.index, y=var.values, marker_color=SERIES[0],
                               hovertemplate="%{x}: %{y:.1f}%<extra></extra>"))
        fig.update_yaxes(title="Variance explained (%)")
        show(style(fig, 340, f"PCA: PC1 (market factor) explains {var.iloc[0]:.0f}% of all moves", legend=False))

    symbols = ["circle", "square", "diamond", "triangle-up", "x", "star"]
    fig = go.Figure()
    for i, cname in enumerate(sorted(clusters.unique())):
        idx = clusters.index[clusters == cname]
        fig.add_trace(go.Scatter(x=feat.loc[idx, "Ann. volatility (%)"], y=feat.loc[idx, "Beta"], mode="markers",
                                 name=cname, text=idx,
                                 marker=dict(size=11, color=SERIES[i], symbol=symbols[i]),
                                 hovertemplate="%{text}<br>Volatility %{x:.1f}%<br>Beta %{y:.2f}<extra>" + cname + "</extra>"))
    fig.update_xaxes(title="Annualised volatility (%)")
    fig.update_yaxes(title="Beta to Nifty")
    show(style(fig, 520, "Stock clusters by risk behaviour"))

    t = feat.copy()
    t.insert(0, "Cluster", clusters)
    t.insert(1, "Sector", [D.STOCKS[s + ".NS"] for s in t.index])
    st.dataframe(t.sort_values("Cluster").style.format("{:.2f}", subset=feat.columns), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        l2 = load["PC2"].sort_values()
        fig = go.Figure(go.Bar(x=l2.values, y=l2.index, orientation="h",
                               marker_color=[SERIES[7] if v < 0 else SERIES[0] for v in l2.values],
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_xaxes(title="PC2 loading")
        show(style(fig, max(420, 16 * len(l2)), "PC2: the second hidden factor (often a sector split)", legend=False))
    with c2:
        Z = (feat - feat.mean()) / feat.std()
        dend = ff.create_dendrogram(Z.values, labels=Z.index.tolist(), orientation="left",
                                    colorscale=SERIES[:8])
        show(style(dend, max(420, 16 * len(Z)), "Hierarchical clustering dendrogram", legend=False))

elif page == "6. Prediction & portfolio":
    st.title("Phase 5: Regime-aware prediction and portfolio")
    c = st.columns(4)
    top_n = c[0].slider("Stocks in portfolio", 4, 12, 8)
    cap = c[1].slider("Max weight per stock (%)", 10, 50, 25) / 100
    cost = c[2].slider("Transaction cost (bps)", 0, 50, 10)
    risk = c[3].toggle("Cut exposure in Bear/Panic", True)

    res = get_models(LIVE, START, TICKERS, test_start)
    st.subheader("Model comparison on the test period")
    st.caption(f"Training: up to {test_start:%d %b %Y}. Test: {test_start:%d %b %Y} onwards. "
               "Target: 5-day forward return. Regimes in the test period use only past data (no look-ahead).")
    st.dataframe(res["metrics"].rename_axis("Model").reset_index().style.format(
        {"Directional accuracy (%)": "{:.1f}", "Information coefficient": "{:.3f}", "RMSE (%)": "{:.2f}"}),
        width="stretch", hide_index=True)

    imp = res["importance"]
    fig = go.Figure(go.Heatmap(z=imp.values, x=imp.columns, y=imp.index, colorscale=SEQUENTIAL,
                               texttemplate="%{z:.0f}", colorbar=dict(title="% of gain"),
                               hovertemplate="%{y} in %{x} model: %{z:.1f}%<extra></extra>"))
    show(style(fig, 400, "Which signals matter in each regime (LightGBM feature importance)", legend=False))
    with st.expander("Feature meanings"):
        st.markdown("`r1`, `r5`, `r21`: past 1/5/21-day return · `vol21`: 21-day volatility · `rsi14`: RSI · "
                    "`dist_ma50`: distance from 50-day average · `vol_z`: volume spike score · "
                    "`mkt_r5`: Nifty 5-day return · `vix`, `vix_chg5`: India VIX level and 5-day change")

    st.subheader("Backtest")
    rets_bt, hist = P.backtest(close, res["test"], stocks, top_n, cap, cost, risk_control=risk)
    eq = (1 + rets_bt).cumprod() * 100
    fig = go.Figure()
    for i, cname in enumerate(eq.columns):
        fig.add_trace(go.Scatter(x=eq.index, y=eq[cname], name=cname, line=dict(width=2, color=SERIES[i]),
                                 hovertemplate="%{x|%d %b %Y}: %{y:.1f}<extra>" + cname + "</extra>"))
    fig.update_yaxes(title="Value of Rs 100")
    show(style(fig, 420, "Growth of Rs 100"))
    table(P.perf(rets_bt))
    st.caption("Weekly rebalancing (every 5 trading days), mean-variance weights, risk-free rate 6% for Sharpe.")
    dd = eq / eq.cummax() - 1
    fig = go.Figure()
    for i, cname in enumerate(dd.columns):
        fig.add_trace(go.Scatter(x=dd.index, y=dd[cname] * 100, name=cname, line=dict(width=2, color=SERIES[i])))
    fig.update_yaxes(title="Drawdown (%)")
    show(style(fig, 320, "Drawdowns"))

    st.subheader("Current recommended portfolio")
    asof, reg_now, port = P.current_portfolio(close, vol, stocks, res, top_n, cap)
    port = port[port["Weight (%)"] > 0.5].sort_values("Weight (%)", ascending=False)
    c = st.columns(3)
    c[0].metric("As of", f"{asof:%d %b %Y}")
    c[1].metric("Regime", reg_now)
    c[2].metric("Invested (rest in cash)", f"{port['Weight (%)'].sum():.0f}%")
    c1, c2 = st.columns([2, 3])
    with c1:
        st.dataframe(port.style.format({"Predicted 5-day return (%)": "{:.2f}", "Weight (%)": "{:.1f}"}),
                     width="stretch", hide_index=True)
    with c2:
        fig = go.Figure(go.Bar(x=port["Weight (%)"], y=port["Stock"], orientation="h", marker_color=SERIES[0],
                               hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="Weight (%)")
        show(style(fig, 320, legend=False))
    st.caption("Educational output from a statistical model. Not investment advice.")

elif page == "7. Key insights":
    st.title("Key insights")
    st.caption("Generated from the current data and settings. Use these sentences in your report.")
    tests, adf, params, gvol, vs, fii = get_stats(LIVE, START, TICKERS, fii_bytes)
    prof = A.regime_profile(labels, nr)
    cr = A.corr_by_regime(rets, labels)
    res = get_models(LIVE, START, TICKERS, test_start)
    rets_bt, _ = P.backtest(close, res["test"], stocks)
    pf = P.perf(rets_bt)
    rs = A.return_stats(nr)
    m = res["metrics"]

    def p_of(h):
        row = tests[tests["Hypothesis"] == h]
        return row["p-value"].iloc[0] if len(row) else None

    out = []
    out.append(f"**Fat tails.** Nifty daily returns have excess kurtosis of {rs['Excess kurtosis']:.1f}; normality is "
               f"{'rejected' if rs['Jarque-Bera p-value'] < 0.05 else 'not rejected'} (Jarque-Bera p {pfmt(rs['Jarque-Bera p-value'])}).")
    if "Bull" in prof.index and "Panic" in prof.index:
        out.append(f"**Regimes.** The market spent {prof.loc['Bull', 'Share of days (%)']:.0f}% of days in Bull and "
                   f"{prof.loc['Panic', 'Share of days (%)']:.0f}% in Panic. Panic episodes last about "
                   f"{prof.loc['Panic', 'Avg duration (days)']:.0f} days on average, with annualised volatility of "
                   f"{prof.loc['Panic', 'Ann. volatility (%)']:.0f}% vs {prof.loc['Bull', 'Ann. volatility (%)']:.0f}% in Bull.")
    if "Panic" in cr and "Bull" in cr:
        out.append(f"**Diversification.** Average correlation between stocks rises from {cr['Bull']:.2f} in Bull to "
                   f"{cr['Panic']:.2f} in Panic, so diversification protects least when it is needed most.")
    p = p_of("India VIX changes help predict Nifty returns")
    out.append(f"**Fear gauge.** India VIX changes {'do' if p < 0.05 else 'do not'} Granger-cause Nifty returns (p {pfmt(p)}).")
    p = p_of("FII net flows help predict Nifty returns")
    if p is not None:
        out.append(f"**FII flows.** FII net flows {'do' if p < 0.05 else 'do not'} help predict Nifty returns (p {pfmt(p)}).")
    out.append(f"**Volume spikes.** Next-day returns after volume-spike up-days average "
               f"{vs['Avg next-day return after spike (%)']:.3f}% vs {vs['Avg next-day return otherwise (%)']:.3f}% "
               f"otherwise (p {pfmt(vs['p'])}).")
    out.append(f"**Volatility memory.** GARCH persistence is {params.get('alpha[1]', 0) + params.get('beta[1]', 0):.2f}, "
               "so volatility shocks fade slowly.")
    out.append(f"**Prediction.** Regime-aware models reach {m.loc['Regime-aware models', 'Directional accuracy (%)']:.1f}% "
               f"directional accuracy vs {m.loc['Single global model', 'Directional accuracy (%)']:.1f}% for one global model "
               f"(information coefficient {m.loc['Regime-aware models', 'Information coefficient']:.3f} vs "
               f"{m.loc['Single global model', 'Information coefficient']:.3f}).")
    s, n = pf.loc["RegimeX strategy"], pf.loc["Nifty 50 buy-and-hold"]
    out.append(f"**Portfolio.** In the backtest the strategy returned {s['CAGR (%)']:.1f}% a year with a "
               f"{s['Max drawdown (%)']:.1f}% max drawdown, vs {n['CAGR (%)']:.1f}% and {n['Max drawdown (%)']:.1f}% "
               f"for Nifty 50 (Sharpe {s['Sharpe ratio']:.2f} vs {n['Sharpe ratio']:.2f}).")
    for line in out:
        st.markdown(f"- {line}")
    if used.startswith("Demo"):
        st.warning("These insights come from demo data. Switch to live data in the sidebar for real findings.")
