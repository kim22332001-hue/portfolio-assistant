# app.py
# Portfolio Assistant (DB version for Streamlit Cloud + Supabase)
# - Tabs: Home / Data / Recommendation v2 / Explain v2 / My Portfolio
# - holdings & trades: stored in Supabase(Postgres) via st.secrets["db"]["url"]
# - prices: fetched from yfinance and cached in-memory (session + st.cache_data)
# NOTE: Remove any parquet-based persistence to ensure cross-device consistency.

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
from sqlalchemy import create_engine, text


# =========================
# Ticker descriptions
# =========================
TICKER_INFO = {
    "SPY": {"name": "SPDR S&P 500 ETF", "desc": "Tracks the S&P 500, representing the overall U.S. equity market"},
    "QQQ": {"name": "Invesco QQQ Trust", "desc": "Tracks NASDAQ-100, focused on technology and growth stocks"},
    "SMH": {"name": "VanEck Semiconductor ETF", "desc": "Tracks global semiconductor manufacturers and designers"},
    "ITA": {"name": "iShares U.S. Aerospace & Defense ETF", "desc": "Tracks U.S. aerospace and defense companies"},
    "PAVE": {"name": "Global X U.S. Infrastructure ETF", "desc": "Tracks U.S. infrastructure development companies"},
    "REMX": {"name": "VanEck Rare Earth ETF", "desc": "Tracks rare earth and strategic metals companies"},
    "XME": {"name": "SPDR Metals & Mining ETF", "desc": "Tracks metals and mining companies"},
    "QTUM": {"name": "Defiance Quantum ETF", "desc": "Tracks quantum computing and next-gen technology companies"},
    "GRID": {"name": "First Trust Smart Grid ETF", "desc": "Tracks smart grid and power infrastructure companies"},
}

DEFAULT_TICKERS = ["SPY", "QQQ", "SMH", "ITA", "PAVE", "REMX", "XME", "QTUM", "GRID"]


# =========================
# Settings (embedded) + Shock SB allowance
# =========================
def default_settings():
    """
    Self-contained settings.
    Added: shock_sb_min/max to allow limited sector allocation (5~10%) during SHOCK if Strong sectors exist.
    """
    return {
        "core": {"sp500_ticker": "SPY"},
        "tickers": {
            "sectors_1x": ["SMH", "ITA", "PAVE", "REMX", "XME", "QTUM", "GRID"],
            "benchmarks": ["QQQ"],
        },
        "allocator_v2": {
            "cash_floor_shock": 0.35,
            "cash_floor_riskon": 0.05,
            "cash_floor_riskoff": 0.20,
            "sb_shock": [0.00, 0.00],          # legacy (kept)
            "sb_riskon": [0.15, 0.35],
            "sb_neutral": [0.05, 0.20],

            # ✅ NEW: allow small SB during SHOCK only for Strong sectors (active>0)
            "shock_sb_min": 0.05,
            "shock_sb_max": 0.10,

            "caps": {
                "SMH": 0.20,
                "ITA": 0.15,
                "PAVE": 0.15,
                "REMX": 0.10,
                "XME": 0.12,
                "QTUM": 0.12,
                "GRID": 0.12,
            },
        },
        "correlation": {
            "pairs": [
                ["SMH", "QQQ"],
                ["XME", "PAVE"],
                ["REMX", "XME"],
                ["SMH", "SPY"],
                ["QQQ", "SPY"],
            ],
            "window": 60,
            "high_threshold": 0.75,
            "cash_floor_bump": 0.10,
            "cap_tighten_pct": 0.15,
        },
    }


# =========================
# DB (Supabase Postgres)
# =========================
@st.cache_resource
def get_engine():
    try:
        db_url = st.secrets["db"]["url"]
    except Exception:
        raise RuntimeError("DB url not set. Add [db].url in Streamlit Secrets.")
    return create_engine(db_url, pool_pre_ping=True)


def init_db():
    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text("""
            create table if not exists holdings (
                ticker text primary key,
                qty double precision not null default 0
            );
        """))
        conn.execute(text("""
            create table if not exists trades (
                id bigserial primary key,
                date date not null,
                ticker text not null,
                action text not null,
                qty double precision not null default 0,
                price double precision not null default 0
            );
        """))
        conn.execute(text("create index if not exists idx_trades_date on trades(date);"))


def load_holdings() -> pd.DataFrame:
    init_db()
    eng = get_engine()
    with eng.begin() as conn:
        rows = conn.execute(text("select ticker, qty from holdings order by ticker;")).fetchall()
    if not rows:
        return pd.DataFrame([{"ticker": "SPY", "qty": 0.0}, {"ticker": "QQQ", "qty": 0.0}])
    return pd.DataFrame(rows, columns=["ticker", "qty"])


def save_holdings(df: pd.DataFrame) -> None:
    init_db()
    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0)

    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text("delete from holdings;"))
        for _, r in df.iterrows():
            t = str(r["ticker"]).strip()
            if t:
                conn.execute(
                    text("insert into holdings(ticker, qty) values(:t, :q);"),
                    {"t": t, "q": float(r["qty"])},
                )


def load_trades() -> pd.DataFrame:
    init_db()
    eng = get_engine()
    with eng.begin() as conn:
        rows = conn.execute(text("""
            select date, ticker, action, qty, price
            from trades
            order by date, ticker, action;
        """)).fetchall()

    if not rows:
        return pd.DataFrame([{"date": "2025-01-02", "ticker": "SPY", "action": "BUY", "qty": 0.0, "price": 0.0}])

    df = pd.DataFrame(rows, columns=["date", "ticker", "action", "qty", "price"])
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def save_trades(df: pd.DataFrame) -> None:
    init_db()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Trades has invalid 'date'. Use YYYY-MM-DD.")

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["action"] = df["action"].astype(str).str.upper().str.strip()
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce").fillna(0.0)
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)

    allowed = {"BUY", "SELL"}
    if not set(df["action"].unique()).issubset(allowed):
        raise ValueError("Trades 'action' must be BUY or SELL.")

    eng = get_engine()
    with eng.begin() as conn:
        conn.execute(text("delete from trades;"))
        for _, r in df.iterrows():
            t = str(r["ticker"]).strip()
            a = str(r["action"]).strip()
            if t and a in allowed:
                conn.execute(
                    text("""
                        insert into trades(date, ticker, action, qty, price)
                        values(:d, :t, :a, :q, :p);
                    """),
                    {
                        "d": r["date"].date(),
                        "t": t,
                        "a": a,
                        "q": float(r["qty"]),
                        "p": float(r["price"]),
                    },
                )


# =========================
# Basic utils
# =========================
def last_date(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "N/A"
    try:
        return str(pd.to_datetime(df.index).max().date())
    except Exception:
        return "N/A"


def fetch_prices(tickers: list[str], period: str = "max") -> pd.DataFrame:
    tickers = [t.strip().upper() for t in tickers if str(t).strip()]
    if not tickers:
        return pd.DataFrame()

    df = yf.download(
        tickers=tickers,
        period=period,
        auto_adjust=True,
        progress=False,
    )["Close"]

    if isinstance(df, pd.Series):
        df = df.to_frame()

    df = df.dropna(how="all")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.sort_index()


@st.cache_data(show_spinner=False, ttl=6 * 3600)
def cached_prices(tickers_tuple: tuple[str, ...], period: str) -> pd.DataFrame:
    return fetch_prices(list(tickers_tuple), period=period)


def safe_series(prices: pd.DataFrame, col: str) -> pd.Series:
    if prices is None or prices.empty or col not in prices.columns:
        return pd.Series(dtype=float)
    return prices[col].dropna()


def pct_change(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()


# =========================
# Indicators
# =========================
def mom_n(prices: pd.Series, n: int) -> float:
    if len(prices) < n + 1:
        return np.nan
    return float(prices.iloc[-1] / prices.iloc[-(n + 1)] - 1)


def rs_n(prices_a: pd.Series, prices_b: pd.Series, n: int) -> float:
    if len(prices_a) < n + 1 or len(prices_b) < n + 1:
        return np.nan
    ra = prices_a.iloc[-1] / prices_a.iloc[-(n + 1)] - 1
    rb = prices_b.iloc[-1] / prices_b.iloc[-(n + 1)] - 1
    return float(ra - rb)


def shock_flag(returns: pd.Series, w: int = 20, k: float = 1.8) -> int:
    if len(returns) < max(w, 6):
        return 0
    vol = returns.iloc[-w:].std()
    r5 = (1 + returns.iloc[-5:]).prod() - 1
    if vol <= 0 or np.isnan(vol):
        return 0
    return int(abs(r5) > k * vol)


def trend_above_200(prices: pd.Series) -> int:
    if len(prices) < 200:
        return 0
    return int(prices.iloc[-1] > prices.iloc[-200:].mean())


# =========================
# Sector signals
# =========================
def build_sector_table(prices: pd.DataFrame, sectors: list[str], sp500: str) -> pd.DataFrame:
    out = []
    sp = safe_series(prices, sp500)

    for s in sectors:
        p = safe_series(prices, s)
        r = pct_change(p)

        m20 = mom_n(p, 20)
        t60 = mom_n(p, 60)
        rs20 = rs_n(p, sp, 20)
        sh = shock_flag(r)

        if sh == 1 or (not np.isnan(t60) and t60 < 0):
            signal = -1
            strength = "Weak"
        elif (not np.isnan(m20) and m20 > 0) and (not np.isnan(rs20) and rs20 > 0):
            signal = 1
            strength = "Strong" if (not np.isnan(t60) and t60 > 0) else "Normal"
        else:
            signal = 0
            strength = "Normal"

        out.append(
            {
                "sector": s,
                "signal": signal,
                "strength": strength,
                "rs20": rs20,
                "mom20": m20,
                "trend60": t60,
                "shock": sh,
            }
        )

    return pd.DataFrame(out).set_index("sector")


# =========================
# Correlation
# =========================
def avg_corr(prices: pd.DataFrame, pairs: list[list[str]], window: int = 60) -> float:
    corrs = []
    for a, b in pairs:
        if a not in prices.columns or b not in prices.columns:
            continue
        ra = pct_change(prices[a])
        rb = pct_change(prices[b])
        df = pd.concat([ra, rb], axis=1).dropna()
        if len(df) >= window:
            corrs.append(df.iloc[-window:].corr().iloc[0, 1])
    if not corrs:
        return np.nan
    return float(np.mean(corrs))


# =========================
# Allocator v2 (FINAL) + Shock SB allowance
# =========================
def allocator_v2(prices: pd.DataFrame, settings: dict) -> tuple[pd.DataFrame, dict]:
    core = settings["core"]["sp500_ticker"]
    sectors = settings["tickers"]["sectors_1x"]
    alloc = settings["allocator_v2"]
    corr_cfg = settings["correlation"]

    spy = safe_series(prices, core)
    qqq = safe_series(prices, "QQQ")

    shock = shock_flag(pct_change(spy))
    risk_on = int(
        shock == 0
        and trend_above_200(spy) == 1
        and (not np.isnan(rs_n(qqq, spy, 20)) and rs_n(qqq, spy, 20) > 0)
    )

    sec = build_sector_table(prices, sectors, core)

    corr_val = avg_corr(prices, corr_cfg["pairs"], corr_cfg["window"])
    corr_high = int(not np.isnan(corr_val) and corr_val >= corr_cfg["high_threshold"])

    if shock:
        cash = float(alloc["cash_floor_shock"])
        sb_min, sb_max = alloc["sb_shock"]
    elif risk_on:
        cash = float(alloc["cash_floor_riskon"])
        sb_min, sb_max = alloc["sb_riskon"]
    else:
        cash = float(alloc["cash_floor_riskoff"])
        sb_min, sb_max = alloc["sb_neutral"]

    if corr_high and not shock:
        cash = min(0.6, cash + float(corr_cfg["cash_floor_bump"]))

    sec["active"] = ((sec["signal"] == 1) & (sec["shock"] == 0)).astype(int)
    active_cnt = int(sec["active"].sum())

    if active_cnt == 0:
        sb = 0.0
    else:
        if shock:
            shock_min = float(alloc.get("shock_sb_min", 0.05))
            shock_max = float(alloc.get("shock_sb_max", 0.10))
            scale = min(1.0, (active_cnt - 1) / 3)
            sb = shock_min + (shock_max - shock_min) * scale
        else:
            scale = min(1.0, (active_cnt - 1) / 4)
            sb = float(sb_min + (sb_max - sb_min) * scale)

    sp500_w = max(0.0, 1.0 - cash - sb)

    sec["score"] = 0.0
    for s in sec.index:
        if int(sec.loc[s, "active"]) == 1:
            sec.loc[s, "score"] = (
                max(0.0, float(sec.loc[s, "rs20"])) * 0.5
                + max(0.0, float(sec.loc[s, "mom20"])) * 0.3
                + max(0.0, float(sec.loc[s, "trend60"])) * 0.2
            )

    if float(sec["score"].sum()) > 0 and sb > 0:
        sec["target"] = sb * sec["score"] / float(sec["score"].sum())
    else:
        sec["target"] = 0.0

    caps = alloc["caps"]
    tighten = float(corr_cfg["cap_tighten_pct"]) if corr_high else 0.0
    sec["cap"] = [float(caps.get(s, 0.0)) * (1.0 - tighten) for s in sec.index]
    sec["target"] = sec[["target", "cap"]].min(axis=1)

    overflow = sb - float(sec["target"].sum())
    if overflow > 0:
        sp500_w += overflow

    summary = {
        "cash": float(cash),
        "sp500": float(sp500_w),
        "sectors": float(sec["target"].sum()),
        "risk_on": int(risk_on),
        "shock": int(shock),
        "corr": float(corr_val) if not np.isnan(corr_val) else np.nan,
        "corr_high": int(corr_high),
        "active_cnt": int(active_cnt),
        "sb": float(sb),
        "shock_sb_min": float(alloc.get("shock_sb_min", 0.05)),
        "shock_sb_max": float(alloc.get("shock_sb_max", 0.10)),
    }

    return sec.reset_index(), summary


# =========================
# Sidebar helpers
# =========================
def sidebar_badges(risk_on: int, shock: int, corr_high: int, corr_val: float | float("nan")) -> None:
    def pill(text: str, bg: str, fg: str = "white"):
        st.sidebar.markdown(
            f"<span style='display:inline-block;padding:4px 10px;border-radius:999px;"
            f"background:{bg};color:{fg};font-size:12px;margin-right:6px;margin-bottom:6px'>{text}</span>",
            unsafe_allow_html=True,
        )

    if shock == 1:
        pill("SHOCK (reduce risk / keep cash)", "#b00020")
    else:
        pill("NO SHOCK (normal conditions)", "#2e7d32")

    if risk_on == 1:
        pill("RISK-ON (add winners / allow SB)", "#1565c0")
    else:
        pill("RISK-OFF (be defensive / favor SPY+cash)", "#6d4c41")

    if corr_high == 1:
        pill("CORR HIGH (avoid concentration)", "#7b1fa2")
    else:
        pill("CORR OK (diversification works)", "#455a64")

    if corr_val is not None and not (isinstance(corr_val, float) and np.isnan(corr_val)):
        st.sidebar.caption(f"Avg corr (pairs): {corr_val:.2f}")
    else:
        st.sidebar.caption("Avg corr (pairs): N/A")


def sidebar_allocation_summary(spy_w: float, cash_w: float, sec_df: pd.DataFrame) -> None:
    spy_pct = spy_w * 100
    cash_pct = cash_w * 100

    st.sidebar.subheader("Recommended Allocation (v2)")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("SPY", f"{spy_pct:.1f}%")
    c2.metric("Cash", f"{cash_pct:.1f}%")

    st.sidebar.markdown("**Other Index (Top 3)**")

    if sec_df is None or sec_df.empty:
        st.sidebar.info("No sector allocation")
        st.sidebar.metric("Others (sum)", "0.0%")
        return

    tmp = sec_df[["sector", "target"]].copy()
    tmp["target"] = pd.to_numeric(tmp["target"], errors="coerce").fillna(0.0)
    tmp = tmp.sort_values("target", ascending=False)

    top3 = tmp.head(3).copy()
    others = tmp.iloc[3:].copy()

    for _, r in top3.iterrows():
        tkr = str(r["sector"]).upper()
        pct = float(r["target"]) * 100
        full = TICKER_INFO.get(tkr, {}).get("name", "")
        if full:
            st.sidebar.write(f"- **{tkr}** ({full}): {pct:.2f}%")
        else:
            st.sidebar.write(f"- **{tkr}**: {pct:.2f}%")

    others_sum = float(others["target"].sum()) * 100 if len(others) > 0 else 0.0
    st.sidebar.write(f"- **Others (sum)**: {others_sum:.2f}%")


# =========================
# Explain v2 helper
# =========================
def render_explain_v2(settings: dict, summary: dict, sec_df: pd.DataFrame) -> None:
    alloc = settings["allocator_v2"]
    corr_cfg = settings["correlation"]

    st.subheader("How this recommendation was calculated (v2)")

    st.markdown(
        """
This tab explains **exactly how today's weights were produced**.

The algorithm has 3 layers:

1) **Market regime** (Risk-on / Risk-off / Shock) decided from SPY & QQQ  
2) **Cash floor + Sector bucket size (SB)** decided from the regime (+ correlation adjustment)  
3) **Sector selection + weights** decided from sector signals and capped to manage risk
"""
    )

    st.divider()
    st.markdown("### 1) Market regime logic")
    st.markdown(
        """
- **Shock = 1** if SPY had an unusually large move over last 5 trading days relative to recent volatility  
- **Risk-on = 1** if:
  - Shock is 0, AND
  - SPY is above its 200-day moving average, AND
  - QQQ has positive 20-day relative strength vs SPY (QQQ outperforms SPY)
- Otherwise **Risk-off / Neutral**
"""
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("shock", int(summary.get("shock", 0)))
    c2.metric("risk_on", int(summary.get("risk_on", 0)))
    c3.metric("corr_high", int(summary.get("corr_high", 0)))

    corr_val = summary.get("corr", np.nan)
    if not (isinstance(corr_val, float) and np.isnan(corr_val)):
        st.caption(f"Avg corr (pairs/window={corr_cfg['window']}): {corr_val:.2f}")
    else:
        st.caption("Avg corr: N/A")

    st.divider()
    st.markdown("### 2) Cash floor + Sector bucket (SB)")
    st.markdown(
        f"""
**Base parameters** (from settings):

- **Shock:** cash = `{alloc['cash_floor_shock']:.2f}`  
- **Risk-on:** cash = `{alloc['cash_floor_riskon']:.2f}`, SB range = `{alloc['sb_riskon'][0]:.2f} ~ {alloc['sb_riskon'][1]:.2f}`  
- **Risk-off:** cash = `{alloc['cash_floor_riskoff']:.2f}`, SB range = `{alloc['sb_neutral'][0]:.2f} ~ {alloc['sb_neutral'][1]:.2f}`  

**Correlation adjustment:** if `corr_high=1` (but not shock), cash is increased by `{corr_cfg['cash_floor_bump']:.2f}` (max 0.60).

✅ **Shock allowance (new):** if Shock=1 AND there are Strong sectors (`active_cnt>0`),  
we still allow **SB = {alloc['shock_sb_min']:.2f} ~ {alloc['shock_sb_max']:.2f}** (5~10%) to keep small “winner” exposure.
"""
    )

    st.markdown("**Today's result:**")
    d1, d2, d3 = st.columns(3)
    d1.metric("Cash", f"{float(summary.get('cash', 0.0))*100:.1f}%")
    d2.metric("Sector bucket (SB)", f"{float(summary.get('sb', 0.0))*100:.1f}%")
    d3.metric("SPY (SP500)", f"{float(summary.get('sp500', 0.0))*100:.1f}%")

    st.divider()
    st.markdown("### 3) Sector signals + weights")
    st.markdown(
        """
For each sector ETF, we compute:

- **mom20**: 20-day momentum  
- **trend60**: 60-day momentum  
- **rs20**: 20-day relative strength vs SPY  
- **shock**: sector shock flag (same concept as SPY shock)

Then:
- If sector shock=1 OR trend60<0 → **signal = -1 (Weak)**
- Else if mom20>0 AND rs20>0 → **signal = +1 (Strong/Active)**
- Else → **signal = 0 (Neutral)**

Only **Active** sectors (signal=1 and shock=0) can receive allocation.

Weights are proportional to a **score**:
- score = 0.5*max(0, rs20) + 0.3*max(0, mom20) + 0.2*max(0, trend60)

Then we apply a **cap per sector**.  
If correlation is high, caps are tightened by `cap_tighten_pct`.  
Any leftover weight from caps flows back into SPY.
"""
    )

    if sec_df is None or sec_df.empty:
        st.info("No sector table available.")
        return

    view = sec_df.copy()
    view["ETF Name"] = view["sector"].map(lambda x: TICKER_INFO.get(str(x).upper(), {}).get("name", ""))
    view["target_%"] = (pd.to_numeric(view["target"], errors="coerce") * 100).round(2)

    cols = ["sector", "ETF Name", "signal", "strength", "rs20", "mom20", "trend60", "shock", "active", "score", "cap", "target_%"]
    for col in cols:
        if col not in view.columns:
            view[col] = np.nan
    view = view[cols].sort_values("target_%", ascending=False)

    st.markdown("**Today's sector table (inputs → signal → target)**")
    st.dataframe(view, use_container_width=True)


# =========================
# Portfolio calc (Trades-based)
# =========================
def normalize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker", "action", "qty"}
    if not required.issubset(set(trades.columns)):
        raise ValueError(f"Trades must include columns: {sorted(required)}")

    t = trades.copy().dropna(how="all")
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    if t["date"].isna().any():
        raise ValueError("Trades has invalid 'date'. Use YYYY-MM-DD.")

    t["ticker"] = t["ticker"].astype(str).str.upper().str.strip()
    t["action"] = t["action"].astype(str).str.upper().str.strip()
    t["qty"] = pd.to_numeric(t["qty"], errors="coerce")

    if t["ticker"].eq("").any():
        raise ValueError("Trades has empty ticker.")
    if t["qty"].isna().any():
        raise ValueError("Trades has invalid qty.")
    if not set(t["action"].unique()).issubset({"BUY", "SELL"}):
        raise ValueError("Trades 'action' must be BUY or SELL.")

    t["signed_qty"] = np.where(t["action"] == "BUY", t["qty"], -t["qty"])
    return t.sort_values("date")


def positions_from_trades(trades: pd.DataFrame, price_index: pd.DatetimeIndex) -> pd.DataFrame:
    t = normalize_trades(trades)
    daily = t.groupby(["date", "ticker"])["signed_qty"].sum().unstack(fill_value=0)
    daily = daily.reindex(price_index, fill_value=0)
    return daily.cumsum()


def portfolio_value_from_positions(positions: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    common = [c for c in positions.columns if c in prices.columns]
    if not common:
        return pd.Series(dtype="float64")
    v = (positions[common] * prices[common]).sum(axis=1)
    v.name = "PortfolioValue"
    return v


# =========================
# UI
# =========================
def banner(asof: str) -> None:
    st.markdown("**Base Currency:** USD")
    st.markdown(f"**Data as of (US close):** `{asof}`" if asof and asof != "N/A" else "**Data as of:** None")


def main():
    st.set_page_config(page_title="Portfolio Assistant", layout="wide")
    settings = default_settings()

    core = settings["core"]["sp500_ticker"]
    sectors = settings["tickers"]["sectors_1x"]
    benchmarks = settings["tickers"]["benchmarks"]
    tickers = list(dict.fromkeys([core] + sectors + benchmarks))

    st.sidebar.header("Settings")
    period = st.sidebar.selectbox("Price history period", ["1y", "2y", "5y", "10y", "max"], index=4)
    st.sidebar.divider()

    st.title("Portfolio Assistant")

    if "prices" not in st.session_state:
        st.session_state["prices"] = pd.DataFrame()

    prices: pd.DataFrame = st.session_state.get("prices", pd.DataFrame())

    c1, c2 = st.columns([2, 1])
    with c1:
        banner(last_date(prices))
    with c2:
        if st.button("Update prices now"):
            with st.spinner("Downloading prices..."):
                prices = fetch_prices(tickers, period=period)
                st.session_state["prices"] = prices
            st.success(f"Updated: {last_date(prices)}")

    # If empty, try cached download automatically (first load convenience)
    if prices is None or prices.empty:
        try:
            prices = cached_prices(tuple(tickers), period=period)
            st.session_state["prices"] = prices
        except Exception:
            prices = pd.DataFrame()

    asof = last_date(prices)

    sec_df = pd.DataFrame()
    summary = {
        "sp500": 0.0,
        "cash": 0.0,
        "sectors": 0.0,
        "risk_on": 0,
        "shock": 0,
        "corr": np.nan,
        "corr_high": 0,
        "active_cnt": 0,
        "sb": 0.0,
        "shock_sb_min": settings["allocator_v2"]["shock_sb_min"],
        "shock_sb_max": settings["allocator_v2"]["shock_sb_max"],
    }
    if prices is not None and not prices.empty:
        try:
            sec_df, summary = allocator_v2(prices, settings)
        except Exception:
            pass

    sidebar_badges(summary.get("risk_on", 0), summary.get("shock", 0), summary.get("corr_high", 0), summary.get("corr", np.nan))
    sidebar_allocation_summary(float(summary.get("sp500", 0.0)), float(summary.get("cash", 0.0)), sec_df)

    tabs = st.tabs(["Home", "Data", "Recommendation v2", "Explain v2", "My Portfolio"])

    with tabs[0]:
        st.write("1) Update prices")
        st.write("2) Open Recommendation v2")
        st.write("3) Explain v2 shows why it recommended those weights")
        st.write("4) Input trades in My Portfolio (saved to DB)")
        if prices.empty:
            st.info("Prices not loaded yet. Click **Update prices now**.")
        else:
            st.success(f"Prices loaded. As of: {asof}")
            st.dataframe(prices.tail(5), use_container_width=True)

    with tabs[1]:
        if prices.empty:
            st.info("No data yet")
        else:
            st.dataframe(prices.tail(20), use_container_width=True)

    with tabs[2]:
        if prices.empty:
            st.warning("Update prices first")
            st.stop()

        st.subheader("Allocation")
        a, b, c = st.columns(3)
        a.metric("SP500", f"{float(summary['sp500'])*100:.1f}%")
        b.metric("Sectors", f"{float(summary['sectors'])*100:.1f}%")
        c.metric("Cash", f"{float(summary['cash'])*100:.1f}%")

        st.caption(
            f"risk_on={summary['risk_on']} | shock={summary['shock']} | "
            f"corr={summary['corr'] if not np.isnan(summary['corr']) else 'NaN'} | "
            f"corr_high={summary['corr_high']} | active_cnt={summary['active_cnt']} | sb={summary['sb']:.3f}"
        )

        st.divider()
        st.subheader("Sector detail (Signals + Targets)")
        show = sec_df.copy()
        if not show.empty:
            show["ETF Name"] = show["sector"].map(lambda x: TICKER_INFO.get(str(x).upper(), {}).get("name", ""))
            show["Description"] = show["sector"].map(lambda x: TICKER_INFO.get(str(x).upper(), {}).get("desc", ""))
            show["target_%"] = (pd.to_numeric(show["target"], errors="coerce") * 100).round(2)

            cols = ["sector", "ETF Name", "Description", "signal", "strength", "rs20", "mom20", "trend60", "shock", "active", "score", "cap", "target_%"]
            for col in cols:
                if col not in show.columns:
                    show[col] = np.nan
            show = show[cols].sort_values("target_%", ascending=False)
            st.dataframe(show, use_container_width=True)
        else:
            st.info("No sector table available. Update prices first.")

    with tabs[3]:
        if prices.empty:
            st.warning("Update prices first")
            st.stop()
        render_explain_v2(settings, summary, sec_df)

    with tabs[4]:
        st.subheader("My Portfolio")
        st.caption("PC/모바일 어디서든 입력 → Supabase(DB)에 저장 → 동일 데이터로 동기화")

        # Load from DB once per session
        if "holdings_df" not in st.session_state:
            st.session_state["holdings_df"] = load_holdings().reset_index(drop=True)

        if "trades_df" not in st.session_state:
            st.session_state["trades_df"] = load_trades().reset_index(drop=True)

        st.markdown("### 1) Current Holdings (Optional)")
        holdings_view = st.session_state["holdings_df"].copy()
        holdings_view["ETF Name"] = holdings_view["ticker"].map(lambda x: TICKER_INFO.get(str(x).upper(), {}).get("name", ""))
        holdings_view["Description"] = holdings_view["ticker"].map(lambda x: TICKER_INFO.get(str(x).upper(), {}).get("desc", ""))

        st.caption("입력/수정은 ticker, qty만 하세요. (Name/Description은 자동 표시)")
        holdings_edited = st.data_editor(
            holdings_view,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "ticker": st.column_config.SelectboxColumn("ticker", options=DEFAULT_TICKERS),
                "qty": st.column_config.NumberColumn("qty", min_value=0.0, step=1.0),
                "ETF Name": st.column_config.TextColumn("ETF Name", disabled=True),
                "Description": st.column_config.TextColumn("Description", disabled=True),
            },
            key="holdings_editor",
        )

        if st.button("Save holdings"):
            save_df = holdings_edited[["ticker", "qty"]].copy()
            save_holdings(save_df)
            st.session_state["holdings_df"] = save_df.reset_index(drop=True)
            st.success("Holdings saved to DB!")

        st.markdown("### 2) Trades History (Recommended)")
        trades_edited = st.data_editor(
            st.session_state["trades_df"],
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "date": st.column_config.TextColumn("date (YYYY-MM-DD)"),
                "ticker": st.column_config.SelectboxColumn("ticker", options=DEFAULT_TICKERS),
                "action": st.column_config.SelectboxColumn("action", options=["BUY", "SELL"]),
                "qty": st.column_config.NumberColumn("qty", min_value=0.0, step=1.0),
                "price": st.column_config.NumberColumn("price", min_value=0.0, step=0.01),
            },
            key="trades_editor",
        )

        if st.button("Save trades"):
            try:
                save_trades(trades_edited.copy())
                st.session_state["trades_df"] = trades_edited.copy().reset_index(drop=True)
                st.success("Trades saved to DB!")
            except Exception as e:
                st.error(f"Save trades failed: {e}")

        st.divider()
        st.markdown("### 3) Valuation & Performance")

        if prices.empty:
            st.warning("먼저 Home에서 **Update prices now**를 눌러 가격 데이터를 받아오세요.")
            st.stop()

        holdings = st.session_state["holdings_df"].copy()
        trades = st.session_state["trades_df"].copy()

        holdings["ticker"] = holdings["ticker"].astype(str).str.upper().str.strip()
        holdings["qty"] = pd.to_numeric(holdings["qty"], errors="coerce").fillna(0.0)

        latest = prices.dropna(how="all").iloc[-1].to_dict() if not prices.empty else {}

        hv_rows = []
        total_value = 0.0
        for _, r in holdings.iterrows():
            tkr = r["ticker"]
            qty = float(r["qty"])
            if qty <= 0:
                continue
            px = latest.get(tkr)
            if px is None or pd.isna(px):
                continue
            val = qty * float(px)
            total_value += val
            hv_rows.append({"ticker": tkr, "qty": qty, "last_px": float(px), "value": val})

        hv = pd.DataFrame(hv_rows).sort_values("value", ascending=False) if hv_rows else pd.DataFrame()

        cva, cvb = st.columns([1, 2])
        with cva:
            st.metric("Holdings value (now)", f"{total_value:,.2f}")
        with cvb:
            st.caption("※ Holdings 기반 현재 평가금액입니다. (현금/수수료/세금은 아직 반영 X)")

        if not hv.empty:
            hv_show = hv.copy()
            hv_show["value"] = hv_show["value"].round(2)
            hv_show["last_px"] = hv_show["last_px"].round(4)
            st.dataframe(hv_show, use_container_width=True)
        else:
            st.info("Holdings에 보유수량을 입력하면 현재 평가표가 나옵니다.")

        st.markdown("#### Trades-based performance vs SPY")
        try:
            tnorm = trades.copy().dropna(how="all")
            if len(tnorm) == 0:
                st.info("Trades를 입력하면 거래 기반 성과 곡선을 계산할 수 있어요.")
                st.stop()

            pos = positions_from_trades(tnorm, prices.index)
            port_val = portfolio_value_from_positions(pos, prices).dropna()

            # Need at least 2 points to draw a curve
            if len(port_val) < 2 or (port_val <= 0).all():
                st.info("Trades 기반 성과를 계산할 수 있는 구간이 부족합니다. (가격 데이터 기간/거래일을 확인)")
                st.stop()

            spy = prices["SPY"] if "SPY" in prices.columns else fetch_prices(["SPY"], period=period)["SPY"]
            spy = spy.reindex(port_val.index).dropna()

            if len(spy) < 2:
                st.info("SPY 비교 구간이 부족합니다. period를 늘리고 Update prices를 다시 해보세요.")
                st.stop()

            # align
            comp = pd.DataFrame({"Portfolio": port_val, "SPY": spy}).dropna()
            if len(comp) < 2:
                st.info("정렬 후 비교 구간이 부족합니다. period를 늘려주세요.")
                st.stop()

            port_norm = comp["Portfolio"] / comp["Portfolio"].iloc[0]
            spy_norm = comp["SPY"] / comp["SPY"].iloc[0]
            out = pd.DataFrame({"Portfolio": port_norm, "SPY": spy_norm}).dropna()

            st.line_chart(out)

            total_return_port = (out["Portfolio"].iloc[-1] - 1) * 100
            total_return_spy = (out["SPY"].iloc[-1] - 1) * 100
            m1, m2, m3 = st.columns(3)
            m1.metric("Portfolio Total Return (%)", f"{total_return_port:.2f}")
            m2.metric("SPY Total Return (%)", f"{total_return_spy:.2f}")
            m3.metric("Alpha vs SPY (pp)", f"{(total_return_port - total_return_spy):.2f}")

            st.caption("※ 현금흐름/수수료/세금 포함(TWR/MWR)은 다음 단계에서 추가 가능합니다.")

        except Exception as e:
            st.error(f"Performance calculation error: {e}")


if __name__ == "__main__":
    main()
