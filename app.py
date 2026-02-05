# app.py
# Portfolio Assistant (Clean Rebuild)
# - Tabs: Home / Data / Recommendation v2 / Explain v2 / My Portfolio
# - Shock에서도 Strong 섹터 SB 5~10% 허용 (확정사항 반영)
# - Sidebar: regime badges (+ action guidance) + allocation summary (SPY/Cash/Top3/Others)
# - Manual holdings & trades input + robust performance vs SPY (no out-of-bounds)
# - Local cache: data/prices.parquet, data/holdings.parquet, data/trades.parquet

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st


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
# Local files
# =========================
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

PRICES_FILE = DATA_DIR / "prices.parquet"
HOLDINGS_FILE = DATA_DIR / "holdings.parquet"
TRADES_FILE = DATA_DIR / "trades.parquet"


# =========================
# Settings (embedded)
# =========================
def default_settings():
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

            # legacy ranges (kept)
            "sb_shock": [0.00, 0.00],
            "sb_riskon": [0.15, 0.35],
            "sb_neutral": [0.05, 0.20],

            # ✅ Shock에서도 Strong 섹터 SB 5~10% 허용
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
# IO utils
# =========================
def safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def safe_write_parquet(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception as e:
        st.warning(f"Failed to save cache: {path.name} ({e})")


def last_date_from_prices(prices: pd.DataFrame) -> str:
    if prices is None or prices.empty:
        return "N/A"
    try:
        return str(pd.to_datetime(prices.index).max().date())
    except Exception:
        return "N/A"


# =========================
# Prices (robust)
# =========================
def fetch_prices(tickers: list[str], period: str = "max") -> pd.DataFrame:
    tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
    tickers = list(dict.fromkeys(tickers))
    if not tickers:
        return pd.DataFrame()

    df = yf.download(
        tickers=tickers,
        period=period,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    # Normalize to close prices
    close = None
    if isinstance(df.columns, pd.MultiIndex):
        # typical multi: ("Close", ticker)
        if "Close" in df.columns.get_level_values(0):
            close = df["Close"].copy()
        else:
            # fallback: try xs
            try:
                close = df.xs("Close", axis=1, level=0, drop_level=True)
            except Exception:
                close = None
    else:
        # single ticker might return columns like Open/High/Low/Close...
        if "Close" in df.columns:
            close = df[["Close"]].copy()
            close.columns = [tickers[0]]
        else:
            # sometimes df itself is a Series-like
            close = df.copy()

    if close is None:
        return pd.DataFrame()

    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])

    # Ensure columns are tickers if single col but mismatched
    if close.shape[1] == 1 and close.columns[0] not in tickers:
        close.columns = [tickers[0]]

    close = close.dropna(how="all")
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.sort_index()


def safe_series(prices: pd.DataFrame, col: str) -> pd.Series:
    if prices is None or prices.empty or col not in prices.columns:
        return pd.Series(dtype=float)
    return prices[col].dropna()


def pct_change(s: pd.Series) -> pd.Series:
    return s.pct_change().dropna()


# =========================
# Indicators
# =========================
def mom_n(prices: pd.Series, n: int) -> float:
    if prices is None or len(prices) < n + 1:
        return np.nan
    return float(prices.iloc[-1] / prices.iloc[-(n + 1)] - 1)


def rs_n(prices_a: pd.Series, prices_b: pd.Series, n: int) -> float:
    if prices_a is None or prices_b is None or len(prices_a) < n + 1 or len(prices_b) < n + 1:
        return np.nan
    ra = prices_a.iloc[-1] / prices_a.iloc[-(n + 1)] - 1
    rb = prices_b.iloc[-1] / prices_b.iloc[-(n + 1)] - 1
    return float(ra - rb)


def shock_flag(returns: pd.Series, w: int = 20, k: float = 1.8) -> int:
    if returns is None or len(returns) < max(w, 6):
        return 0
    vol = returns.iloc[-w:].std()
    r5 = (1 + returns.iloc[-5:]).prod() - 1
    if vol <= 0 or np.isnan(vol):
        return 0
    return int(abs(r5) > k * vol)


def trend_above_200(prices: pd.Series) -> int:
    if prices is None or len(prices) < 200:
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
    rel = rs_n(qqq, spy, 20)
    risk_on = int(shock == 0 and trend_above_200(spy) == 1 and (not np.isnan(rel) and rel > 0))

    sec = build_sector_table(prices, sectors, core)

    corr_val = avg_corr(prices, corr_cfg["pairs"], corr_cfg["window"])
    corr_high = int(not np.isnan(corr_val) and corr_val >= corr_cfg["high_threshold"])

    # Base cash floor / SB range by regime
    if shock:
        cash = float(alloc["cash_floor_shock"])
        sb_min, sb_max = alloc["sb_shock"]
    elif risk_on:
        cash = float(alloc["cash_floor_riskon"])
        sb_min, sb_max = alloc["sb_riskon"]
    else:
        cash = float(alloc["cash_floor_riskoff"])
        sb_min, sb_max = alloc["sb_neutral"]

    # corr high -> more cash (but not during shock)
    if corr_high and not shock:
        cash = min(0.6, cash + float(corr_cfg["cash_floor_bump"]))

    # Active = Strong(+1) and not sector-shock
    sec["active"] = ((sec["signal"] == 1) & (sec["shock"] == 0)).astype(int)
    active_cnt = int(sec["active"].sum())

    # Sector bucket size (SB)
    if active_cnt == 0:
        sb = 0.0
    else:
        if shock:
            # ✅ Shock에서도 Strong 섹터 SB 5~10% 허용
            shock_min = float(alloc.get("shock_sb_min", 0.05))
            shock_max = float(alloc.get("shock_sb_max", 0.10))
            scale = min(1.0, (active_cnt - 1) / 3)  # cap at 4 actives
            sb = shock_min + (shock_max - shock_min) * scale
        else:
            scale = min(1.0, (active_cnt - 1) / 4)
            sb = float(sb_min + (sb_max - sb_min) * scale)

    sp500_w = max(0.0, 1.0 - cash - sb)

    # score active sectors
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

    # Caps (+ tighten when corr high)
    caps = alloc["caps"]
    tighten = float(corr_cfg["cap_tighten_pct"]) if corr_high else 0.0
    sec["cap"] = [float(caps.get(s, 0.0)) * (1.0 - tighten) for s in sec.index]

    # enforce cap
    sec["target"] = sec[["target", "cap"]].min(axis=1)

    # overflow goes to SP500
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
# Sidebar UI helpers
# =========================
def pill(text: str, bg: str, fg: str = "white"):
    st.sidebar.markdown(
        f"<span style='display:inline-block;padding:4px 10px;border-radius:999px;"
        f"background:{bg};color:{fg};font-size:12px;margin-right:6px;margin-bottom:6px'>{text}</span>",
        unsafe_allow_html=True,
    )


def sidebar_badges(risk_on: int, shock: int, corr_high: int, corr_val: float) -> None:
    # ✅ 배지에 행동가이드(괄호) 포함
    if shock == 1:
        pill("SHOCK (리스크 축소 / 현금↑, Strong만 5~10%)", "#b00020")
    else:
        pill("NO SHOCK (일반 규칙 적용)", "#2e7d32")

    if risk_on == 1:
        pill("RISK-ON (승자추세 / SB 허용)", "#1565c0")
    else:
        pill("RISK-OFF (방어 / SPY+Cash 선호)", "#6d4c41")

    if corr_high == 1:
        pill("CORR HIGH (집중↓ / 캡 타이트)", "#7b1fa2")
    else:
        pill("CORR OK (분산 효과 양호)", "#455a64")

    if corr_val is not None and not (isinstance(corr_val, float) and np.isnan(corr_val)):
        st.sidebar.caption(f"Avg corr: {corr_val:.2f}")
    else:
        st.sidebar.caption("Avg corr: N/A")


def sidebar_allocation_summary(spy_w: float, cash_w: float, sec_df: pd.DataFrame) -> None:
    st.sidebar.subheader("Recommended Allocation (v2)")
    c1, c2 = st.sidebar.columns(2)
    c1.metric("SPY", f"{spy_w*100:.1f}%")
    c2.metric("Cash", f"{cash_w*100:.1f}%")

    st.sidebar.markdown("**Other Index (Top 3)**")

    if sec_df is None or sec_df.empty:
        st.sidebar.info("No sector allocation")
        st.sidebar.metric("Others (sum)", "0.0%")
        return

    tmp = sec_df[["sector", "target"]].copy()
    tmp["target"] = pd.to_numeric(tmp["target"], errors="coerce").fillna(0.0)
    tmp = tmp.sort_values("target", ascending=False)

    top3 = tmp.head(3)
    others = tmp.iloc[3:]

    for _, r in top3.iterrows():
        tkr = str(r["sector"]).upper()
        pct = float(r["target"]) * 100
        full = TICKER_INFO.get(tkr, {}).get("name", "")
        if full:
            st.sidebar.write(f"- **{tkr}** ({full}): {pct:.2f}%")
        else:
            st.sidebar.write(f"- **{tkr}**: {pct:.2f}%")

    others_sum = float(others["target"].sum()) * 100 if len(others) else 0.0
    st.sidebar.write(f"- **Others (sum)**: {others_sum:.2f}%")


# =========================
# Explain tab
# =========================
def render_explain_v2(settings: dict, summary: dict, sec_df: pd.DataFrame) -> None:
    alloc = settings["allocator_v2"]
    corr_cfg = settings["correlation"]

    st.subheader("How this recommendation was calculated (v2)")

    st.markdown(
        """
이 탭은 **오늘 추천 비중이 왜 이렇게 나왔는지**를 그대로 설명합니다.

레이어는 3개:

1) **Market regime** (Risk-on / Risk-off / Shock)  
2) **Cash floor + Sector bucket size(SB)** (Shock에서는 Strong 섹터만 5~10% 예외 허용)  
3) **Sector signals + capped weights**
"""
    )

    st.divider()
    st.markdown("### 1) Market regime")
    st.markdown(
        """
- **Shock=1**: SPY가 최근 5거래일 변동이 최근 변동성 대비 과도하면 Shock
- **Risk-on=1** 조건:
  - Shock=0
  - SPY > 200일 평균
  - QQQ가 SPY 대비 20일 상대강도(rs20) 양수
- 그 외는 **Risk-off**
"""
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("shock", int(summary.get("shock", 0)))
    c2.metric("risk_on", int(summary.get("risk_on", 0)))
    c3.metric("corr_high", int(summary.get("corr_high", 0)))

    corr_val = summary.get("corr", np.nan)
    if not (isinstance(corr_val, float) and np.isnan(corr_val)):
        st.caption(f"Avg corr (window={corr_cfg['window']}): {corr_val:.2f}")
    else:
        st.caption("Avg corr: N/A")

    st.divider()
    st.markdown("### 2) Cash floor + Sector bucket (SB)")
    st.markdown(
        f"""
기본 파라미터:

- Shock: cash = `{alloc['cash_floor_shock']:.2f}`
- Risk-on: cash = `{alloc['cash_floor_riskon']:.2f}`, SB = `{alloc['sb_riskon'][0]:.2f} ~ {alloc['sb_riskon'][1]:.2f}`
- Risk-off: cash = `{alloc['cash_floor_riskoff']:.2f}`, SB = `{alloc['sb_neutral'][0]:.2f} ~ {alloc['sb_neutral'][1]:.2f}`

상관 높음(corr_high=1)이고 Shock가 아니면 cash를 `{corr_cfg['cash_floor_bump']:.2f}` 만큼 추가(최대 0.60)

✅ **Shock 예외(확정):** Shock=1이라도 Strong 섹터(active>0)가 있으면  
SB를 **{alloc['shock_sb_min']:.2f} ~ {alloc['shock_sb_max']:.2f} (5~10%)** 범위로 허용
"""
    )

    d1, d2, d3 = st.columns(3)
    d1.metric("Cash", f"{float(summary.get('cash', 0.0))*100:.1f}%")
    d2.metric("Sector bucket (SB)", f"{float(summary.get('sb', 0.0))*100:.1f}%")
    d3.metric("SPY", f"{float(summary.get('sp500', 0.0))*100:.1f}%")

    st.divider()
    st.markdown("### 3) Sector signals + weights")
    st.markdown(
        """
각 섹터 ETF별로:

- mom20(20일 모멘텀), trend60(60일 모멘텀), rs20(SPY 대비 20일 상대강도), sector-shock

룰:
- sector-shock=1 또는 trend60<0 → Weak
- mom20>0 & rs20>0 → Active(Strong/Normal)
- 그 외 Neutral

Active만 SB를 배정하고,
score = 0.5*rs20 + 0.3*mom20 + 0.2*trend60(각각 0 이하 절삭)로 비중 배분  
이후 cap 적용(상관 높으면 cap 더 타이트), 남는 비중은 SPY로 환류
"""
    )

    if sec_df is None or sec_df.empty:
        st.info("No sector table available.")
        return

    view = sec_df.copy()
    view["ETF Name"] = view["sector"].map(lambda x: TICKER_INFO.get(str(x).upper(), {}).get("name", ""))
    view["target_%"] = (pd.to_numeric(view["target"], errors="coerce").fillna(0.0) * 100).round(2)

    cols = ["sector", "ETF Name", "signal", "strength", "rs20", "mom20", "trend60", "shock", "active", "score", "cap", "target_%"]
    for col in cols:
        if col not in view.columns:
            view[col] = np.nan
    view = view[cols].sort_values("target_%", ascending=False)

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
    t["qty"] = pd.to_numeric(t["qty"], errors="coerce").fillna(0.0)

    if (t["ticker"] == "").any():
        raise ValueError("Trades has empty ticker.")
    if not set(t["action"].unique()).issubset({"BUY", "SELL"}):
        raise ValueError("Trades 'action' must be BUY or SELL.")
    if (t["qty"] < 0).any():
        raise ValueError("Trades qty must be >= 0.")

    t["signed_qty"] = np.where(t["action"] == "BUY", t["qty"], -t["qty"])
    t = t.sort_values("date")
    return t


def positions_from_trades(trades: pd.DataFrame, price_index: pd.DatetimeIndex) -> pd.DataFrame:
    t = normalize_trades(trades)
    daily = t.groupby(["date", "ticker"])["signed_qty"].sum().unstack(fill_value=0.0)
    # align to price index and accumulate
    daily = daily.reindex(pd.to_datetime(price_index), fill_value=0.0)
    return daily.cumsum()


def portfolio_value_from_positions(positions: pd.DataFrame, prices: pd.DataFrame) -> pd.Series:
    common = [c for c in positions.columns if c in prices.columns]
    if not common:
        return pd.Series(dtype="float64")
    v = (positions[common] * prices[common]).sum(axis=1)
    v.name = "PortfolioValue"
    return v


def safe_performance_curve(trades_df: pd.DataFrame, prices: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Returns comp dataframe with columns [Portfolio, SPY] normalized to 1.0 at start.
    Always guards against empty/out-of-bounds.
    """
    if trades_df is None or trades_df.dropna(how="all").empty:
        raise ValueError("Trades is empty.")

    pos = positions_from_trades(trades_df, prices.index)
    port_val = portfolio_value_from_positions(pos, prices)

    port_val = port_val.replace([np.inf, -np.inf], np.nan).dropna()
    # 거래 전 구간(port=0)을 제거해서 정규화 오류 방지
    port_val = port_val[port_val > 0]
    if len(port_val) < 2:
        raise ValueError("No valid portfolio value segment (check trades/prices).")

    # SPY series
    if "SPY" in prices.columns:
        spy = prices["SPY"].dropna()
    else:
        spy_df = fetch_prices(["SPY"], period=period)
        if spy_df.shape[1] == 1 and "SPY" not in spy_df.columns:
            spy_df.columns = ["SPY"]
        spy = spy_df["SPY"].dropna()

    idx = port_val.index.intersection(spy.index)
    if len(idx) < 2:
        raise ValueError("Not enough overlapping dates between Portfolio and SPY.")

    port_val = port_val.reindex(idx)
    spy = spy.reindex(idx)

    port_norm = port_val / float(port_val.iloc[0])
    spy_norm = spy / float(spy.iloc[0])

    comp = pd.DataFrame({"Portfolio": port_norm, "SPY": spy_norm}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(comp) < 2:
        raise ValueError("Comparison series became empty after cleaning.")
    return comp


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

    # Sidebar controls
    st.sidebar.header("Settings")
    period = st.sidebar.selectbox("Price history period", ["1y", "2y", "5y", "10y", "max"], index=4)
    st.sidebar.divider()

    # Load cached prices into session
    if "prices" not in st.session_state:
        cached = safe_read_parquet(PRICES_FILE)
        if not cached.empty and "date" in cached.columns:
            cached["date"] = pd.to_datetime(cached["date"])
            cached = cached.set_index("date")
        st.session_state["prices"] = cached

    prices: pd.DataFrame = st.session_state.get("prices", pd.DataFrame())
    asof = last_date_from_prices(prices)

    # Title + update
    st.title("Portfolio Assistant")
    c1, c2 = st.columns([2, 1])
    with c1:
        banner(asof)
    with c2:
        if st.button("Update prices now"):
            with st.spinner("Downloading prices..."):
                prices = fetch_prices(tickers, period=period)
                st.session_state["prices"] = prices
                safe_write_parquet(prices.reset_index(names="date"), PRICES_FILE)
                asof = last_date_from_prices(prices)
            st.success(f"Updated: {asof}")

    # Precompute recommendation when prices exist
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

    # Sidebar rendering
    sidebar_badges(int(summary.get("risk_on", 0)), int(summary.get("shock", 0)), int(summary.get("corr_high", 0)), summary.get("corr", np.nan))
    sidebar_allocation_summary(float(summary.get("sp500", 0.0)), float(summary.get("cash", 0.0)), sec_df)

    tabs = st.tabs(["Home", "Data", "Recommendation v2", "Explain v2", "My Portfolio"])

    # Home
    with tabs[0]:
        st.write("1) Update prices")
        st.write("2) Open Recommendation v2")
        st.write("3) Explain v2 shows why it recommended those weights")
        st.write("4) Input trades/holdings in My Portfolio")

        if prices.empty:
            st.info("Prices not loaded yet.")
        else:
            st.success(f"Prices loaded. As of: {asof}")
            st.dataframe(prices.tail(5), use_container_width=True)

    # Data
    with tabs[1]:
        if prices.empty:
            st.info("No data yet")
        else:
            st.dataframe(prices.tail(30), use_container_width=True)

    # Recommendation v2
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
            show["target_%"] = (pd.to_numeric(show["target"], errors="coerce").fillna(0.0) * 100).round(2)

            cols = ["sector", "ETF Name", "Description", "signal", "strength", "rs20", "mom20", "trend60", "shock", "active", "score", "cap", "target_%"]
            for col in cols:
                if col not in show.columns:
                    show[col] = np.nan
            show = show[cols].sort_values("target_%", ascending=False)

            st.dataframe(show, use_container_width=True)
        else:
            st.info("No sector table available. Update prices first.")

    # Explain v2
    with tabs[3]:
        if prices.empty:
            st.warning("Update prices first")
            st.stop()
        render_explain_v2(settings, summary, sec_df)

    # My Portfolio
    with tabs[4]:
        st.subheader("My Portfolio")
        st.caption("CSV 업로드 없이 웹에서 직접 입력 → 저장 → 성과(거래 기반) vs SPY")

        # Init holdings
        if "holdings_df" not in st.session_state:
            h = safe_read_parquet(HOLDINGS_FILE)
            if h.empty:
                h = pd.DataFrame([{"ticker": "SPY", "qty": 0.0}, {"ticker": "QQQ", "qty": 0.0}])
            st.session_state["holdings_df"] = h.reset_index(drop=True)

        # Init trades
        if "trades_df" not in st.session_state:
            t = safe_read_parquet(TRADES_FILE)
            if t.empty:
                t = pd.DataFrame([{"date": "2025-01-02", "ticker": "SPY", "action": "BUY", "qty": 0.0, "price": 0.0}])
            st.session_state["trades_df"] = t.reset_index(drop=True)

        # Holdings editor
        st.markdown("### 1) Current Holdings (Optional)")
        holdings_view = st.session_state["holdings_df"].copy()
        holdings_view["ticker"] = holdings_view["ticker"].astype(str).str.upper().str.strip()
        holdings_view["ETF Name"] = holdings_view["ticker"].map(lambda x: TICKER_INFO.get(str(x).upper(), {}).get("name", ""))
        holdings_view["Description"] = holdings_view["ticker"].map(lambda x: TICKER_INFO.get(str(x).upper(), {}).get("desc", ""))

        st.caption("입력/수정은 ticker, qty만 하세요. (Name/Description 자동 표시)")
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

        colh1, colh2 = st.columns([1, 3])
        with colh1:
            if st.button("Save holdings"):
                save_df = holdings_edited[["ticker", "qty"]].copy()
                save_df["ticker"] = save_df["ticker"].astype(str).str.upper().str.strip()
                save_df["qty"] = pd.to_numeric(save_df["qty"], errors="coerce").fillna(0.0)
                st.session_state["holdings_df"] = save_df
                safe_write_parquet(save_df, HOLDINGS_FILE)
                st.success("Holdings saved!")

        st.divider()

        # Trades editor
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

        colt1, colt2 = st.columns([1, 3])
        with colt1:
            if st.button("Save trades"):
                st.session_state["trades_df"] = trades_edited.copy()
                safe_write_parquet(st.session_state["trades_df"], TRADES_FILE)
                st.success("Trades saved!")

        st.divider()

        # Valuation
        st.markdown("### 3) Valuation & Performance")

        if prices.empty:
            st.warning("먼저 Home에서 **Update prices now**로 가격 데이터를 받아오세요.")
            st.stop()

        # Holdings valuation (optional)
        holdings = st.session_state["holdings_df"].copy()
        holdings["ticker"] = holdings["ticker"].astype(str).str.upper().str.strip()
        holdings["qty"] = pd.to_numeric(holdings["qty"], errors="coerce").fillna(0.0)

        latest_row = prices.dropna(how="all")
        latest = latest_row.iloc[-1].to_dict() if not latest_row.empty else {}

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
            st.caption("※ Holdings 기반 현재 평가금액입니다. (현금/수수료/세금 미반영)")

        if not hv.empty:
            hv_show = hv.copy()
            hv_show["value"] = hv_show["value"].round(2)
            hv_show["last_px"] = hv_show["last_px"].round(4)
            st.dataframe(hv_show, use_container_width=True)
        else:
            st.info("Holdings에 보유수량을 입력하면 현재 평가표가 나옵니다.")

        st.markdown("#### Trades-based performance vs SPY")

        try:
            trades = st.session_state["trades_df"].copy()
            comp = safe_performance_curve(trades, prices, period=period)

            st.line_chart(comp)

            total_return_port = (float(comp["Portfolio"].iloc[-1]) - 1) * 100
            total_return_spy = (float(comp["SPY"].iloc[-1]) - 1) * 100

            m1, m2, m3 = st.columns(3)
            m1.metric("Portfolio Total Return (%)", f"{total_return_port:.2f}")
            m2.metric("SPY Total Return (%)", f"{total_return_spy:.2f}")
            m3.metric("Alpha vs SPY (pp)", f"{(total_return_port - total_return_spy):.2f}")

            st.caption("※ 거래로 인한 보유수량 변화만 반영(현금흐름/수수료/세금 미반영).")

        except Exception as e:
            st.error(f"Performance calculation error: {e}")
            st.info(
                "체크리스트:\n"
                "- trades에 date(YYYY-MM-DD), ticker, action(BUY/SELL), qty가 유효한지\n"
                "- Update prices now로 해당 ticker 가격이 실제로 받아졌는지\n"
                "- 거래 전 기간 포트가 0이면 자동으로 제거되므로, 최소 1회 이상 BUY가 있어야 함"
            )


if __name__ == "__main__":
    main()
