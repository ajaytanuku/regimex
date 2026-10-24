"""Live market feed (Yahoo Finance) and the paper-trading account."""
import sqlite3
import time
from datetime import datetime
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .data import INDEX, VIX, short

IST = ZoneInfo("Asia/Kolkata")
OPEN, CLOSE = dtime(9, 15), dtime(15, 30)


def now_ist():
    return datetime.now(IST)


def in_session_hours(now=None):
    now = now or now_ist()
    return now.weekday() < 5 and OPEN <= now.time() <= CLOSE


def _quotes(close):
    close = close.ffill()
    q = pd.DataFrame({"Price": close.iloc[-1], "Prev close": close.iloc[-2]})
    q["Change"] = q["Price"] - q["Prev close"]
    q["Change %"] = q["Change"] / q["Prev close"] * 100
    return q


def live_feed(tickers):
    """Latest prices and today's 1-minute Nifty chart from Yahoo Finance."""
    import yfinance as yf
    daily = yf.download(list(tickers) + [INDEX, VIX], period="5d", interval="1d",
                        auto_adjust=False, progress=False, threads=True)["Close"].dropna(how="all")
    if daily.empty or daily[INDEX].dropna().empty:
        raise RuntimeError("No live quotes returned by Yahoo Finance")
    intra = yf.download(INDEX, period="5d", interval="1m", auto_adjust=False, progress=False)["Close"]
    if isinstance(intra, pd.DataFrame):
        intra = intra.iloc[:, 0]
    intra = intra.dropna()
    intra.index = intra.index.tz_convert(IST).tz_localize(None)
    intra = intra[intra.index.date == intra.index[-1].date()]
    last_day = daily.index[-1].date()
    is_open = in_session_hours() and last_day == now_ist().date()
    return _quotes(daily), intra, is_open


def demo_feed(tickers, base):
    """Simulated ticking market built on the last daily closes, so the page moves offline."""
    now = now_ist()
    names = list(tickers) + [INDEX, VIX]
    rng = np.random.default_rng(int(now.strftime("%Y%m%d")))
    steps = rng.normal(0, 1, (1440, len(names))) * 0.0005
    mkt = steps[:, len(tickers)]
    steps[:, : len(tickers)] += mkt[:, None]          # stocks co-move with the index
    steps[:, -1] = -4 * mkt + rng.normal(0, 0.001, 1440)  # VIX rises when the index falls
    m = now.hour * 60 + now.minute
    start = max(0, m - 374)                            # a 375-minute session ending now
    path = np.exp(np.cumsum(steps[start:m + 1], axis=0))
    path /= path[0]
    jitter = np.random.default_rng(int(time.time() // 5)).normal(0, 0.0004, len(names))
    path[-1] *= np.exp(jitter)
    b = base.reindex(names).values
    px = pd.DataFrame(path * b, columns=names,
                      index=pd.date_range(end=now.replace(second=0, microsecond=0, tzinfo=None),
                                          periods=len(path), freq="min"))
    close = pd.concat([pd.DataFrame([b], columns=names), px.iloc[[-1]]])
    return _quotes(close), px[INDEX], True


# ---------------- paper trading ----------------
DB = Path(__file__).resolve().parent.parent / "paper_trades.db"
START_CASH = 1_000_000.0
CHARGES = 0.001  # 0.1% brokerage + taxes per order


class Book:
    """One trader's paper-trading account, stored in SQLite."""

    def __init__(self, user, db=DB):
        self.user = (user or "guest").strip().lower() or "guest"
        self.con = sqlite3.connect(db, check_same_thread=False)
        self.con.execute("CREATE TABLE IF NOT EXISTS trades (user TEXT, ts TEXT, stock TEXT, side TEXT, "
                         "qty INTEGER, price REAL, charges REAL, note TEXT)")

    def trades(self):
        return pd.read_sql("SELECT ts, stock, side, qty, price, charges, note FROM trades "
                           "WHERE user = ? ORDER BY rowid", self.con, params=(self.user,))

    def holdings(self):
        """Current positions with average buy price, and cash left."""
        cash, pos = START_CASH, {}
        for t in self.trades().itertuples():
            qty, cost = pos.get(t.stock, (0, 0.0))
            if t.side == "BUY":
                cash -= t.qty * t.price + t.charges
                qty, cost = qty + t.qty, cost + t.qty * t.price
            else:
                cash += t.qty * t.price - t.charges
                cost -= cost / qty * t.qty
                qty -= t.qty
            pos[t.stock] = (qty, cost)
        rows = [{"stock": s, "qty": q, "avg_price": c / q} for s, (q, c) in pos.items() if q > 0]
        return pd.DataFrame(rows, columns=["stock", "qty", "avg_price"]), cash

    def order(self, stock, side, qty, price, note="Manual"):
        qty = int(qty)
        if qty <= 0:
            raise ValueError("Quantity must be at least 1")
        hold, cash = self.holdings()
        charges = qty * price * CHARGES
        if side == "BUY" and qty * price + charges > cash:
            raise ValueError(f"Not enough cash: need Rs {qty * price + charges:,.0f}, have Rs {cash:,.0f}")
        if side == "SELL":
            held = int(hold.loc[hold["stock"] == stock, "qty"].sum())
            if qty > held:
                raise ValueError(f"You hold only {held} shares of {short(stock)}")
        self.con.execute("INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         (self.user, now_ist().strftime("%Y-%m-%d %H:%M:%S"), stock, side, qty,
                          float(price), float(charges), note))
        self.con.commit()

    def reset(self):
        self.con.execute("DELETE FROM trades WHERE user = ?", (self.user,))
        self.con.commit()


def rebalance(book, target, prices, note="RegimeX rebalance"):
    """Trade the account to target weights {ticker: weight}. Sells first, then buys."""
    hold, cash = book.holdings()
    held = dict(zip(hold["stock"], hold["qty"]))
    equity = cash + sum(q * prices[s] for s, q in held.items())
    want = {s: int(w * equity / prices[s]) for s, w in target.items()}
    done = []
    for s, q in held.items():
        diff = q - want.get(s, 0)
        if diff > 0:
            book.order(s, "SELL", diff, prices[s], note)
            done.append(("SELL", s, diff))
    for s, q in want.items():
        diff = q - held.get(s, 0)
        if diff > 0:
            _, cash = book.holdings()
            diff = min(diff, int(cash / (prices[s] * (1 + CHARGES))))
            if diff > 0:
                book.order(s, "BUY", diff, prices[s], note)
                done.append(("BUY", s, diff))
    return done
