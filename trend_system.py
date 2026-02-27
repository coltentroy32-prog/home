"""
Trend + Trade Signal System (Long-Only) with VIX Gate
======================================================
Two-layer system: rf_blend_25 trend signal + VIX>EMA gated trade signal.

Architecture:
    Layer 1 (TREND):  25% walk-forward Random Forest + 75% rules-based V2 score
                      Classifies macro trend as BULLISH / NEUTRAL / BEARISH
    Layer 2 (TRADE):  Short-term momentum/volume signal, GATED by VIX > EMA63
                      Only reduces exposure when volatility is elevated

    VIX Gate:  When VIX <= EMA63 (gate CLOSED), trade signal is ignored —
               stay fully invested if trend is bullish.
               When VIX > EMA63 (gate OPEN), trade signal can reduce to 50%.

    Allocation:
        Trend BULLISH + Gate CLOSED           → 100%
        Trend BULLISH + Gate OPEN + Trade BUL → 100%
        Trend BULLISH + Gate OPEN + Trade BEA → 50%
        Trend NEUTRAL or BEARISH              → 0%

Usage:
    python trend_system.py --tickers AAPL MSFT GOOGL --days 1260

Requirements:
    pip install numpy pandas scipy scikit-learn openpyxl yfinance

Outputs:
    Equity_trend_positions.xlsx with sheets:
        Positions       - All tickers with current signal state
        Alerts          - New Buy or Sell signals today
        Buy             - Today's new Buy signals
        Sell            - Today's new Sell signals
        Hold Long       - Currently long positions (100%)
        Reduced Exposure - Currently reduced positions (50%)
        No Position     - Currently flat (0%)
"""
import sys
import os
import pickle
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# =========================================================================
# CONFIGURATION
# =========================================================================
DEFAULT_TICKERS = [
    "SPY", "QQQ", "IWM", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
    "TSLA", "WMT", "JPM", "V", "UNH", "JNJ", "PCT", "CELH", "MA", "XOM",
    "COF", "REAL", "KO", "PEP", "COST", "RH", "VIK", "AVGO", "IVZ",
    "WRBY", "ULTA", "NKE", "PM", "PLNT", "DKNG", "NFLX", "AMD", "INTC",
    "RCL", "CCL", "JPXN", "GLD", "XLE", "XLC", "XLP", "XLU", "XLRE", "XLK",
    "XLI", "XLB", "XLY", "XLF", "XLV", "NLR", "LMT", "UFO", "QTUM", "COIN",
    "CPER", "GME", "HSY", "SBUX", "TXG", "TJX", "IBIT", "OC",
]

# FRED API key for RVX data (free: https://fred.stlouisfed.org/docs/api/api_key.html)
FRED_API_KEY = os.environ.get("FRED_API_KEY", "1742dc41673795b3e3b5b2958ca6a65a")


# =========================================================================
# DATA LOADING
# =========================================================================

# Map tickers to their most relevant volatility index
VOL_INDEX_MAP = {
    "SPY": "^VIX", "QQQ": "^VXN", "IWM": "^RVX",
}


def _load_vix_series(start_date, end_date):
    """Load VIX as the base vol signal. Used for all US equities."""
    import yfinance as yf
    try:
        raw = yf.download("^VIX", start=start_date.strftime("%Y-%m-%d"),
                          end=end_date.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if not raw.empty:
            return raw["Close"]
    except Exception:
        pass
    return None


def _load_vol_index(vol_ticker, start_date, end_date):
    """Try to load a specific vol index (VIX, VXN, RVX). Falls back gracefully."""
    import yfinance as yf
    try:
        raw = yf.download(vol_ticker, start=start_date.strftime("%Y-%m-%d"),
                          end=end_date.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if not raw.empty and len(raw) > 50:
            return raw["Close"]
    except Exception:
        pass

    # RVX fallback: try FRED API
    if vol_ticker == "^RVX":
        fred_key = FRED_API_KEY
        if fred_key:
            try:
                import requests
                url = "https://api.stlouisfed.org/fred/series/observations"
                params = {
                    "series_id": "RVXCLS",
                    "api_key": fred_key,
                    "file_type": "json",
                    "observation_start": start_date.strftime("%Y-%m-%d"),
                    "observation_end": end_date.strftime("%Y-%m-%d"),
                }
                resp = requests.get(url, params=params, timeout=15)
                if resp.status_code == 200:
                    obs = resp.json().get("observations", [])
                    dates, vals = [], []
                    for o in obs:
                        if o["value"] != ".":
                            dates.append(pd.Timestamp(o["date"]))
                            vals.append(float(o["value"]))
                    if len(dates) > 50:
                        return pd.Series(vals, index=dates, name="Close")
            except Exception:
                pass
    return None


def _compute_blended_vol(df, vix_series):
    """Blend VIX with stock's own realized vol: 0.6 * VIX + 0.4 * RV21."""
    lr = np.log(df["close"] / df["close"].shift(1))
    rv21 = lr.rolling(21).std() * np.sqrt(252) * 100
    if vix_series is not None:
        vix_aligned = vix_series.reindex(df.index, method="ffill")
        blended = 0.6 * vix_aligned.values + 0.4 * rv21.values
        blended = np.where(np.isnan(blended), rv21.values, blended)
        return blended
    else:
        return rv21.values


def load_data(ticker, days=1260, end_date=None):
    """Load OHLCV + volatility data for a single ticker."""
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    buffer = 1600
    total_days = days + buffer
    if end_date is None:
        end_date = datetime.today()
    start_date = end_date - timedelta(days=int(total_days * 1.6))

    try:
        raw = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"),
                          end=end_date.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
        if raw.empty or len(raw) < 300:
            return None
    except Exception:
        return None

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = pd.DataFrame(index=raw.index)
    df["close"] = raw["Close"].values
    df["high"] = raw["High"].values
    df["low"] = raw["Low"].values
    df["open"] = raw["Open"].values
    df["volume"] = raw["Volume"].values.astype(float)

    vol_ticker = VOL_INDEX_MAP.get(ticker, None)
    if vol_ticker:
        vol_series = _load_vol_index(vol_ticker, start_date, end_date)
        if vol_series is not None:
            df["vix"] = vol_series.reindex(df.index, method="ffill").values
        else:
            vix_series = _load_vix_series(start_date, end_date)
            df["vix"] = _compute_blended_vol(df, vix_series)
    else:
        vix_series = _load_vix_series(start_date, end_date)
        df["vix"] = _compute_blended_vol(df, vix_series)

    df.dropna(subset=["close"], inplace=True)
    return df


# =========================================================================
# FEATURE ENGINEERING
# =========================================================================

def compute_hurst(s, ml=50):
    s = np.array(s, dtype=float)
    s = s[~np.isnan(s)]
    if len(s) < ml * 2:
        ml = len(s) // 4
    if ml < 10:
        return 0.5
    rv = []
    for lag in range(10, ml + 1):
        rl = []
        for st in range(0, len(s) - lag, lag // 2):
            sub = s[st : st + lag]
            if len(sub) < lag:
                continue
            d = np.cumsum(sub - np.mean(sub))
            R = max(d) - min(d)
            S = np.std(sub, ddof=1)
            if S > 0:
                rl.append(R / S)
        if rl:
            rv.append((np.log(lag), np.log(np.mean(rl))))
    if len(rv) < 3:
        return 0.5
    sl, _, _, _, _ = stats.linregress(*zip(*rv))
    return np.clip(sl, 0.01, 0.99)


def build_features(df):
    c = df["close"].values
    h = df["high"].values
    lo = df["low"].values
    v = df["volume"].values.astype(float)
    vx = df["vix"].values
    n = len(df)

    for span in [10, 15, 21, 42, 63]:
        df[f"ema_{span}"] = pd.Series(c, index=df.index).ewm(span=span, adjust=False).mean()
    df["sma_126"] = pd.Series(c, index=df.index).rolling(126).mean()
    df["sma_200"] = pd.Series(c, index=df.index).rolling(200).mean()

    for p in [5, 10, 15, 21, 42, 63]:
        df[f"roc_{p}"] = pd.Series(c, index=df.index).pct_change(p)

    df["ema_42_slope"] = df["ema_42"].pct_change(10)
    df["ema_42_accel"] = df["ema_42_slope"].diff(5)
    df["ema_63_slope"] = df["ema_63"].pct_change(10)
    df["roc_21_accel"] = df["roc_21"].diff(10)
    df["roc_42_accel"] = df["roc_42"].diff(15)

    df["range_pos_63"] = (c - pd.Series(lo, index=df.index).rolling(63).min().values) / (
        pd.Series(h, index=df.index).rolling(63).max().values
        - pd.Series(lo, index=df.index).rolling(63).min().values
        + 1e-10
    )
    df["range_pos_21"] = (c - pd.Series(lo, index=df.index).rolling(21).min().values) / (
        pd.Series(h, index=df.index).rolling(21).max().values
        - pd.Series(lo, index=df.index).rolling(21).min().values
        + 1e-10
    )

    df["ema_15_vs_42"] = (df["ema_15"].values - df["ema_42"].values) / (df["ema_42"].values + 1e-10)
    df["ema_15_vs_63"] = (df["ema_15"].values - df["ema_63"].values) / (df["ema_63"].values + 1e-10)
    df["ema_42_vs_126"] = (df["ema_42"].values - df["sma_126"].values) / (df["sma_126"].values + 1e-10)
    df["price_vs_ema42"] = (c - df["ema_42"].values) / (df["ema_42"].values + 1e-10)
    df["price_vs_ema63"] = (c - df["ema_63"].values) / (df["ema_63"].values + 1e-10)
    df["price_vs_sma200"] = (c - df["sma_200"].values) / (df["sma_200"].values + 1e-10)

    df["rvol_63"] = v / (pd.Series(v, index=df.index).rolling(63).mean().values + 1e-10)

    pd_d = np.diff(c, prepend=c[0])
    obv = np.cumsum(np.where(pd_d > 0, v, np.where(pd_d < 0, -v, 0)))
    on = (obv - pd.Series(obv, index=df.index).rolling(63).mean().values) / (
        pd.Series(obv, index=df.index).rolling(63).std().values + 1e-10
    )
    pn = (c - pd.Series(c, index=df.index).rolling(63).mean().values) / (
        pd.Series(c, index=df.index).rolling(63).std().values + 1e-10
    )
    df["obv_divergence"] = on - pn

    df["vwap_63"] = pd.Series(c * v, index=df.index).rolling(63).sum() / (
        pd.Series(v, index=df.index).rolling(63).sum() + 1e-10
    )
    df["vwap_spread"] = (c - df["vwap_63"].values) / (c + 1e-10)

    lr = np.log(c[1:] / c[:-1])
    lr = np.insert(lr, 0, 0)
    df["log_returns"] = lr

    df["rvol_5d"] = pd.Series(lr, index=df.index).rolling(5).std() * np.sqrt(252) * 100
    df["rvol_15d"] = pd.Series(lr, index=df.index).rolling(15).std() * np.sqrt(252) * 100
    df["rvol_63d"] = pd.Series(lr, index=df.index).rolling(63).std() * np.sqrt(252) * 100

    df["vix_ema63"] = pd.Series(vx, index=df.index).ewm(span=63, adjust=False).mean()
    df["vix_vs_trend"] = (vx - df["vix_ema63"].values) / (df["vix_ema63"].values + 1e-10)
    df["vol_regime"] = df["rvol_15d"].values / (df["rvol_63d"].values + 1e-10)
    df["vix_level"] = vx
    df["vix_roc_21"] = pd.Series(vx, index=df.index).pct_change(21)

    hurst = np.full(n, np.nan)
    for i in range(126, n):
        hurst[i] = compute_hurst(lr[max(0, i - 126) : i], 42)
    df["hurst"] = hurst

    for period in [21, 42]:
        er = np.abs(pd.Series(c, index=df.index).diff(period)) / (
            pd.Series(np.abs(np.diff(c, prepend=c[0])), index=df.index).rolling(period).sum() + 1e-10
        )
        fast, slow = 2 / 3, 2 / 31
        sc_a = (er * (fast - slow) + slow) ** 2
        kama = np.full(n, np.nan)
        kama[period] = c[period]
        for j in range(period + 1, n):
            if not np.isnan(sc_a.iloc[j]) and not np.isnan(kama[j - 1]):
                kama[j] = kama[j - 1] + sc_a.iloc[j] * (c[j] - kama[j - 1])
            elif not np.isnan(kama[j - 1]):
                kama[j] = kama[j - 1]
        df[f"kama_{period}"] = kama

    df["kama_42_slope"] = pd.Series(df["kama_42"].values, index=df.index).pct_change(10)
    df["price_vs_kama42"] = (c - df["kama_42"].values) / (df["kama_42"].values + 1e-10)

    v2 = compute_v2_signal(df)
    df["v2_signal"] = v2

    fwd = df["close"].shift(-21) / df["close"] - 1
    df["target"] = np.where(fwd > 0, 1, 0)

    return df


# =========================================================================
# SIGNAL COMPONENTS
# =========================================================================

FEATURE_COLS = [
    "roc_5", "roc_10", "roc_15", "roc_21", "roc_42", "roc_63",
    "ema_42_slope", "ema_42_accel", "ema_63_slope", "roc_21_accel", "roc_42_accel",
    "range_pos_63", "range_pos_21", "ema_15_vs_42", "ema_15_vs_63", "ema_42_vs_126",
    "price_vs_ema42", "price_vs_ema63", "price_vs_sma200",
    "rvol_63", "obv_divergence", "vwap_spread",
    "rvol_5d", "rvol_15d", "rvol_63d", "vix_vs_trend", "vol_regime", "vix_level", "vix_roc_21",
    "hurst", "kama_42_slope", "price_vs_kama42", "v2_signal",
]


def compute_v2_signal(df):
    n = len(df)
    sc = np.full(n, np.nan)
    for i in range(200, n):
        if np.isnan(df["close"].iloc[i]):
            continue
        e15 = df["ema_15"].iloc[i]; e42 = df["ema_42"].iloc[i]; s126 = df["sma_126"].iloc[i]
        if any(np.isnan([e15, e42, s126])):
            continue

        ps = 0
        stk = (1 if e15 > e42 else -1) + (1 if e42 > s126 else -1)
        ps += stk * 15

        r21, r42, r63 = df["roc_21"].iloc[i], df["roc_42"].iloc[i], df["roc_63"].iloc[i]
        if not any(np.isnan([r21, r42, r63])):
            ps += (0.35 * np.sign(r21) + 0.35 * np.sign(r42) + 0.30 * np.sign(r63)) * np.clip(abs(r42) / 0.08, 0, 1) * 30

        a42 = df["ema_42_accel"].iloc[i]
        if not np.isnan(a42):
            ps += np.clip(a42 / 0.0008, -1, 1) * 30

        a21 = df["roc_21_accel"].iloc[i]
        if not np.isnan(a21):
            ps += np.clip(a21 / 0.015, -1, 1) * 20

        rp = df["range_pos_63"].iloc[i]
        if not np.isnan(rp):
            ps += (rp - 0.5) * 15

        ps = np.clip(ps, -100, 100)

        vos = 0
        vrg = df["vol_regime"].iloc[i]
        if not np.isnan(vrg):
            vos -= np.clip((vrg - 1) / 0.25, -1, 1) * 30
        vvt = df["vix_vs_trend"].iloc[i]
        if not np.isnan(vvt):
            vos -= np.clip(vvt / 0.20, -1, 1) * 30
        vos = np.clip(vos, -100, 100)

        fs = 0
        hu = df["hurst"].iloc[i]
        if not np.isnan(hu):
            pers = (hu - 0.5) * 2
            if pers > 0:
                fs = np.sign(ps) * pers * 50
            else:
                fs = -np.sign(ps) * abs(pers) * 40
        fs = np.clip(fs, -100, 100)

        sc[i] = np.clip(0.40 * ps + 0.30 * vos + 0.30 * fs, -100, 100)

    return pd.Series(sc, index=df.index).ewm(span=5, adjust=False).mean()


def walk_forward_rf(df):
    n = len(df)
    probs = np.full(n, np.nan)
    scaler = StandardScaler()

    valid = df[FEATURE_COLS + ["target"]].dropna()
    if len(valid) < 300:
        return probs

    fv = df.index.get_loc(valid.index[0])

    for rp in range(fv + 1260, n - 21, 63):
        tr = df.iloc[max(0, rp - 1260) : rp][FEATURE_COLS + ["target"]].dropna()
        if len(tr) < 200:
            continue
        X = tr[FEATURE_COLS].values
        y = tr["target"].values
        scaler.fit(X)
        Xs = scaler.transform(X)
        m = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=20,
            max_features="sqrt", random_state=42, n_jobs=-1,
        )
        m.fit(Xs, y)
        nxt = min(rp + 63, n)
        pv = df.iloc[rp:nxt][FEATURE_COLS].dropna()
        if len(pv) == 0:
            continue
        Xp = scaler.transform(pv[FEATURE_COLS].values)
        pb = m.predict_proba(Xp)[:, 1]
        for j, idx in enumerate(pv.index):
            probs[df.index.get_loc(idx)] = pb[j]

    return probs


def compute_trade_signal(df):
    n = len(df)
    sc = np.full(n, np.nan)
    for i in range(63, n):
        if np.isnan(df["close"].iloc[i]):
            continue

        ps = 0
        r5, r10, r15 = df["roc_5"].iloc[i], df["roc_10"].iloc[i], df["roc_15"].iloc[i]
        if not any(np.isnan([r5, r10, r15])):
            ps = (0.5 * np.sign(r5) + 0.3 * np.sign(r10) + 0.2 * np.sign(r15)) * np.clip(abs(r5) / 0.03, 0, 1) * 50

        rp = df["range_pos_21"].iloc[i]
        if not np.isnan(rp):
            ps += (rp - 0.5) * 30

        if i >= 5:
            e15 = df["ema_15"].iloc[i]
            e15p = df["ema_15"].iloc[i - 5]
            if not np.isnan(e15) and not np.isnan(e15p) and e15p > 0:
                ps += np.clip((e15 - e15p) / e15p / 0.005, -1, 1) * 20

        ps = np.clip(ps, -100, 100)

        vs = 0
        rv = df["rvol_63"].iloc[i]
        if not np.isnan(rv) and not np.isnan(r5):
            vs = np.clip((rv - 1) / 0.5, -1, 1) * np.sign(r5) * 30
        od = df["obv_divergence"].iloc[i]
        if not np.isnan(od):
            vs += np.clip(od, -2, 2) * 20
        vs = np.clip(vs, -100, 100)

        vos = 0
        vrg = df["vol_regime"].iloc[i]
        if not np.isnan(vrg):
            vos -= np.clip((vrg - 1) / 0.3, -1, 1) * 30
        vvt = df["vix_vs_trend"].iloc[i]
        if not np.isnan(vvt):
            vos -= np.clip(vvt / 0.2, -1, 1) * 20
        vos = np.clip(vos, -100, 100)

        sc[i] = np.clip(0.40 * ps + 0.25 * vs + 0.35 * vos, -100, 100)

    df["trade_signal"] = pd.Series(sc, index=df.index).ewm(span=3, adjust=False).mean()
    return df


def classify_trade(df):
    n = len(df)
    st = ["NEUTRAL"] * n
    cur = "NEUTRAL"
    for i in range(1, n):
        s = df["trade_signal"].iloc[i]
        if np.isnan(s):
            st[i] = cur
            continue
        if cur == "BULLISH":
            if s < -23: cur = "BEARISH"
            elif s < -15: cur = "NEUTRAL"
        elif cur == "BEARISH":
            if s > 23: cur = "BULLISH"
            elif s > 15: cur = "NEUTRAL"
        else:
            if s > 15: cur = "BULLISH"
            elif s < -15: cur = "BEARISH"
        st[i] = cur
    df["trade_state"] = np.array(st)
    return df


# =========================================================================
# SIGNAL ORCHESTRATION
# =========================================================================

def compute_signals(df):
    """Compute all signal components. Returns dict with arrays aligned to df.index.

    Returns:
        trend_state:   BULLISH/NEUTRAL/BEARISH from rf_blend_25
        trade_state:   BULLISH/NEUTRAL/BEARISH from short-term signal
        vix_gate_open: True when VIX > EMA63 (trade signal applies)
        blend_score:   Raw blended trend score
        rf_probs:      Walk-forward RF probabilities
    """
    n = len(df)

    # Layer 1: Trend — rf_blend_25
    rf_probs = walk_forward_rf(df)
    v2_sig = df["v2_signal"].values
    blend = np.where(
        np.isnan(rf_probs), v2_sig,
        0.25 * (rf_probs - 0.5) * 200 + 0.75 * v2_sig,
    )

    # Trend classification with hysteresis
    trend_state = np.array(["NEUTRAL"] * n)
    cur = "NEUTRAL"
    for i in range(1, n):
        s = blend[i] if not np.isnan(blend[i]) else 0
        if cur == "BULLISH":
            if s < -45: cur = "BEARISH"
            elif s < -25: cur = "NEUTRAL"
        elif cur == "BEARISH":
            if s > 45: cur = "BULLISH"
            elif s > 25: cur = "NEUTRAL"
        else:
            if s > 25: cur = "BULLISH"
            elif s < -25: cur = "BEARISH"
        trend_state[i] = cur

    # Layer 2: Trade (already classified in df)
    trade_state = df["trade_state"].values

    # VIX Gate: trade signal only matters when VIX > EMA63
    vix_gate_open = df["vix"].values > df["vix_ema63"].values

    return {
        "trend_state": trend_state,
        "trade_state": trade_state,
        "vix_gate_open": vix_gate_open,
        "blend_score": blend,
        "rf_probs": rf_probs,
    }


def compute_allocations(df, signals=None):
    """Compute daily allocation percentages using VIX-gated logic.
    Fast path for backtesting — no price level computation.

    Returns:
        np.array of allocation fractions (0.0, 0.5, or 1.0) aligned to df.index
    """
    if signals is None:
        signals = compute_signals(df)

    n = len(df)
    trend = signals["trend_state"]
    trade = signals["trade_state"]
    gate = signals["vix_gate_open"]

    allocation = np.zeros(n)
    for i in range(n):
        if trend[i] != "BULLISH":
            allocation[i] = 0.0
        elif not gate[i]:
            # VIX gate CLOSED — ignore trade signal, stay fully invested
            allocation[i] = 1.0
        elif trade[i] == "BULLISH":
            allocation[i] = 1.0
        else:
            # VIX elevated AND trade not bullish — reduce
            allocation[i] = 0.5

    return allocation


# =========================================================================
# PRICE LEVEL COMPUTATION
# =========================================================================

def _trade_price_score(df, i, p):
    """Price-dependent portion of compute_trade_signal, evaluated at price p."""
    ps = 0
    r5 = (p / df["close"].iloc[i - 5] - 1) if i >= 5 and df["close"].iloc[i - 5] > 0 else np.nan
    r10 = (p / df["close"].iloc[i - 10] - 1) if i >= 10 and df["close"].iloc[i - 10] > 0 else np.nan
    r15 = (p / df["close"].iloc[i - 15] - 1) if i >= 15 and df["close"].iloc[i - 15] > 0 else np.nan

    if not any(np.isnan([r5, r10, r15])):
        ps = (0.5 * np.sign(r5) + 0.3 * np.sign(r10) + 0.2 * np.sign(r15)) * np.clip(abs(r5) / 0.03, 0, 1) * 50

    lo21 = df["low"].iloc[max(0, i - 20) : i + 1].min()
    hi21 = df["high"].iloc[max(0, i - 20) : i + 1].max()
    rp = (p - min(lo21, p)) / (max(hi21, p) - min(lo21, p) + 1e-10)
    ps += (rp - 0.5) * 30

    if i >= 5:
        alpha15 = 2 / 16
        e15_now = alpha15 * p + (1 - alpha15) * df["ema_15"].iloc[i - 1]
        e15p = df["ema_15"].iloc[i - 5]
        if not np.isnan(e15p) and e15p > 0:
            ps += np.clip((e15_now - e15p) / e15p / 0.005, -1, 1) * 20

    return np.clip(ps, -100, 100)


def _trend_price_score(df, i, p):
    """Price-dependent portion of compute_v2_signal, evaluated at price p."""
    alpha15, alpha42 = 2 / 16, 2 / 43
    e15 = alpha15 * p + (1 - alpha15) * df["ema_15"].iloc[i - 1]
    e42 = alpha42 * p + (1 - alpha42) * df["ema_42"].iloc[i - 1]
    s126 = df["sma_126"].iloc[i]

    if any(np.isnan([e15, e42, s126])):
        return 0

    ps = 0
    stk = (1 if e15 > e42 else -1) + (1 if e42 > s126 else -1)
    ps += stk * 15

    r21 = (p / df["close"].iloc[i - 21] - 1) if i >= 21 and df["close"].iloc[i - 21] > 0 else np.nan
    r42 = (p / df["close"].iloc[i - 42] - 1) if i >= 42 and df["close"].iloc[i - 42] > 0 else np.nan
    r63 = (p / df["close"].iloc[i - 63] - 1) if i >= 63 and df["close"].iloc[i - 63] > 0 else np.nan

    if not any(np.isnan([r21, r42, r63])):
        ps += (0.35 * np.sign(r21) + 0.35 * np.sign(r42) + 0.30 * np.sign(r63)) * np.clip(abs(r42) / 0.08, 0, 1) * 30

    lo63 = df["low"].iloc[max(0, i - 62) : i + 1].min()
    hi63 = df["high"].iloc[max(0, i - 62) : i + 1].max()
    rp = (p - min(lo63, p)) / (max(hi63, p) - min(lo63, p) + 1e-10)
    ps += (rp - 0.5) * 15

    return np.clip(ps, -100, 100)


def _find_level(df, i, price_score_func, direction):
    """Binary search for price where price_score_func crosses zero."""
    current = df["close"].iloc[i]
    for lookback in [252, 504, 756, i + 1]:
        start_idx = max(0, i - lookback)
        hist_lo = df["low"].iloc[start_idx : i + 1].min()
        hist_hi = df["high"].iloc[start_idx : i + 1].max()

        if direction == "down":
            lo_p, hi_p = hist_lo, current * 1.001
        else:
            lo_p, hi_p = current * 0.999, hist_hi

        val_lo = price_score_func(df, i, lo_p)
        val_hi = price_score_func(df, i, hi_p)

        if (val_lo > 0) != (val_hi > 0):
            for _ in range(100):
                mid = (lo_p + hi_p) / 2
                val = price_score_func(df, i, mid)
                if direction == "down":
                    if val > 0: hi_p = mid
                    else: lo_p = mid
                else:
                    if val < 0: lo_p = mid
                    else: hi_p = mid
                if abs(hi_p - lo_p) / (current + 1e-10) < 0.0001:
                    break
            return (lo_p + hi_p) / 2

    if direction == "down":
        return df["low"].iloc[0 : i + 1].min()
    else:
        return df["high"].iloc[0 : i + 1].max()


def compute_trade_level(df, i):
    current = df["close"].iloc[i]
    current_ps = _trade_price_score(df, i, current)
    regime = "Bullish" if current_ps > 0 else "Bearish"
    if regime == "Bullish":
        level = _find_level(df, i, _trade_price_score, "down")
    else:
        level = _find_level(df, i, _trade_price_score, "up")
    return level, regime


def compute_trend_level(df, i):
    current = df["close"].iloc[i]
    current_ps = _trend_price_score(df, i, current)
    regime = "Bullish" if current_ps > 0 else "Bearish"
    if regime == "Bullish":
        level = _find_level(df, i, _trend_price_score, "down")
    else:
        level = _find_level(df, i, _trend_price_score, "up")
    return level, regime


# =========================================================================
# POSITION TRACKER (with VIX Gate)
# =========================================================================

def compute_positions(df, compute_levels_last_n=1):
    """Run the full two-layer system with VIX gate.

    Args:
        df: DataFrame with features, trade_signal, trade_state already computed
        compute_levels_last_n: only compute price levels for the last N days
                               (0 = skip entirely, faster for backtesting)

    Returns:
        list of history dicts with daily position state
    """
    n = len(df)

    # Get all signal components
    signals = compute_signals(df)
    trend_st = signals["trend_state"]
    trade_st = signals["trade_state"]
    vix_gate = signals["vix_gate_open"]

    start_i = 200
    entry_date = None
    entry_price = None
    prev_alloc = 0.0
    history = []

    # Only compute price levels for the last N days
    levels_start = max(start_i, n - compute_levels_last_n) if compute_levels_last_n > 0 else n + 1

    for i in range(start_i, n):
        if np.isnan(df["close"].iloc[i]):
            continue

        cp = df["close"].iloc[i]
        trend = trend_st[i]
        trade = trade_st[i]
        gate_open = vix_gate[i]

        # Price levels (only for recent days — expensive to compute)
        if i >= levels_start:
            trade_lvl, _ = compute_trade_level(df, i)
            trend_lvl, _ = compute_trend_level(df, i)
        else:
            trade_lvl = np.nan
            trend_lvl = np.nan

        # ── VIX-GATED ALLOCATION LOGIC ──────────────────────────────
        if trend != "BULLISH":
            allocation = 0.0
            position = "Bearish"
        elif not gate_open:
            # Gate CLOSED — VIX calm, ignore trade signal
            allocation = 1.0
            position = "Bullish"
        elif trade == "BULLISH":
            # Gate OPEN but trade still bullish
            allocation = 1.0
            position = "Bullish"
        else:
            # Gate OPEN and trade not bullish — reduce
            allocation = 0.5
            position = "Reduced"
        # ────────────────────────────────────────────────────────────

        # Determine action based on allocation transitions
        action = None
        if prev_alloc == 0 and allocation > 0:
            action = "Buy"
            entry_date = df.index[i]
            entry_price = cp
        elif prev_alloc > 0 and allocation == 0:
            action = "Sell"
            entry_date = None
            entry_price = None
        elif prev_alloc == 1.0 and allocation == 0.5:
            action = "Reduce Exposure"
        elif prev_alloc == 0.5 and allocation == 1.0:
            action = "Increase Exposure"

        # Default hold actions
        if action is None:
            if position == "Bullish":
                action = "Hold Long"
            elif position == "Reduced":
                action = "Reduced Exposure"
            else:
                action = "No Position"

        in_position = allocation > 0

        history.append({
            "date": df.index[i],
            "close": cp,
            "trade_level": round(trade_lvl, 3) if not np.isnan(trade_lvl) else None,
            "trend_level": round(trend_lvl, 3) if not np.isnan(trend_lvl) else None,
            "trade_regime": "Bullish" if trade == "BULLISH" else "Bearish",
            "trend_regime": "Bullish" if trend == "BULLISH" else "Bearish",
            "vix_gate": "Open" if gate_open else "Closed",
            "position": position,
            "allocation": allocation,
            "action": action,
            "entry_date": entry_date if in_position else None,
            "entry_price": entry_price if in_position else None,
        })

        prev_alloc = allocation

    return history


# =========================================================================
# OUTPUT GENERATION
# =========================================================================

def build_output_row(ticker, h):
    """Build a single row for the output spreadsheet from the last history entry."""
    last = h[-1]
    cp = last["close"]
    trade_lvl = last["trade_level"]
    trend_lvl = last["trend_level"]

    close_vs_trade = (cp - trade_lvl) / trade_lvl if trade_lvl and trade_lvl > 0 else 0
    close_vs_trend = (cp - trend_lvl) / trend_lvl if trend_lvl and trend_lvl > 0 else 0

    pos = last["position"]
    action = last["action"]
    in_position = last["allocation"] > 0

    days_in = None
    if last["entry_date"] is not None:
        days_in = (last["date"] - last["entry_date"]).days

    trade_ret = None
    if last["entry_price"] is not None and last["entry_price"] > 0:
        trade_ret = (cp - last["entry_price"]) / last["entry_price"]

    return {
        "Ticker": ticker,
        "Latest_Date": last["date"],
        "Close": round(cp, 3),
        "TRADE_Level": trade_lvl,
        "Close_vs_TRADE_%": close_vs_trade,
        "TRADE_Regime": last["trade_regime"],
        "TREND_Level": trend_lvl,
        "Close_vs_TREND_%": close_vs_trend,
        "TREND_Regime": last["trend_regime"],
        "VIX_Gate": last["vix_gate"],
        "Position_Today": pos,
        "Allocation": last["allocation"],
        "Action_Today": action,
        "Position_Since": last["entry_date"] if in_position else None,
        "Days_in_Position": days_in if in_position else None,
        "Entry_Price": round(last["entry_price"], 2) if last["entry_price"] and in_position else None,
        "Current_Trade_Return": trade_ret if in_position else None,
    }


def write_xlsx(rows, output_path):
    """Write the output Excel file."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment

    wb = Workbook()

    header_fill = PatternFill("solid", fgColor="1F4E79")
    header_font = Font(bold=True, color="FFFFFF", name="Arial", size=10)
    bullish_fill = PatternFill("solid", fgColor="C6EFCE")
    bearish_fill = PatternFill("solid", fgColor="FFC7CE")
    neutral_fill = PatternFill("solid", fgColor="FFEB9C")
    reduced_fill = PatternFill("solid", fgColor="BDD7EE")
    default_font = Font(name="Arial", size=10)

    columns = [
        "Ticker", "Latest_Date", "Close", "TRADE_Level", "Close_vs_TRADE_%",
        "TRADE_Regime", "TREND_Level", "Close_vs_TREND_%", "TREND_Regime",
        "VIX_Gate", "Position_Today", "Allocation", "Action_Today",
        "Position_Since", "Days_in_Position", "Entry_Price", "Current_Trade_Return",
    ]
    col_widths = [10, 14, 12, 14, 16, 14, 14, 16, 14, 12, 14, 12, 18, 14, 16, 12, 18]

    df_all = pd.DataFrame(rows)
    alerts = df_all[df_all["Action_Today"].isin(["Buy", "Sell", "Increase Exposure", "Reduce Exposure"])]
    buys = df_all[df_all["Action_Today"].isin(["Buy", "Increase Exposure"])]
    sells = df_all[df_all["Action_Today"].isin(["Sell", "Reduce Exposure"])]
    hold_long = df_all[df_all["Action_Today"] == "Hold Long"]
    reduced = df_all[df_all["Action_Today"] == "Reduced Exposure"]
    no_pos = df_all[df_all["Action_Today"] == "No Position"]

    sheets_data = [
        ("Positions", df_all),
        ("Alerts", alerts),
        ("Buy", buys),
        ("Sell", sells),
        ("Hold Long", hold_long),
        ("Reduced Exposure", reduced),
        ("No Position", no_pos),
    ]

    for idx, (sheet_name, data) in enumerate(sheets_data):
        if idx == 0:
            ws = wb.active
            ws.title = sheet_name
        else:
            ws = wb.create_sheet(sheet_name)

        for col_i, col_name in enumerate(columns, 1):
            cell = ws.cell(row=1, column=col_i, value=col_name)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center")

        for col_i, w in enumerate(col_widths, 1):
            col_letter = chr(64 + col_i) if col_i <= 26 else "A" + chr(64 + col_i - 26)
            ws.column_dimensions[col_letter].width = w

        for row_i, (_, row) in enumerate(data.iterrows(), 2):
            for col_i, col_name in enumerate(columns, 1):
                val = row.get(col_name)
                cell = ws.cell(row=row_i, column=col_i, value=val)
                cell.font = default_font
                if col_name in ["TRADE_Regime", "TREND_Regime", "Position_Today"]:
                    if val == "Bullish":
                        cell.fill = bullish_fill
                    elif val == "Bearish":
                        cell.fill = bearish_fill
                    elif val == "Neutral":
                        cell.fill = neutral_fill
                    elif val == "Reduced":
                        cell.fill = reduced_fill
                if col_name == "VIX_Gate":
                    if val == "Closed":
                        cell.fill = bullish_fill
                    elif val == "Open":
                        cell.fill = neutral_fill
                if col_name in ["Close_vs_TRADE_%", "Close_vs_TREND_%", "Current_Trade_Return"]:
                    if val is not None:
                        cell.number_format = "0.0%"
                if col_name in ["Latest_Date", "Position_Since"]:
                    if val is not None:
                        cell.number_format = "YYYY-MM-DD"
                if col_name in ["Close", "TRADE_Level", "TREND_Level", "Entry_Price"]:
                    if val is not None:
                        cell.number_format = "#,##0.00"
                if col_name == "Allocation":
                    if val is not None:
                        cell.number_format = "0%"

        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:{chr(64 + len(columns))}{max(2, len(data) + 1)}"

    wb.save(output_path)
    print(f"  Saved: {output_path}")
    return output_path


# =========================================================================
# CACHING
# =========================================================================

CACHE_DIR = "trend_cache"


def _cache_path(ticker):
    return os.path.join(CACHE_DIR, f"{ticker}_ohlcv.pkl")


def _load_cached_ohlcv(ticker):
    path = _cache_path(ticker)
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        df = data["df"]
        return df, df.index[-1]
    except Exception:
        return None, None


def _save_ohlcv_cache(ticker, df):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = _cache_path(ticker)
    base_cols = ["close", "high", "low", "open", "volume", "vix"]
    with open(path, "wb") as f:
        pickle.dump({"df": df[base_cols].copy(), "saved": datetime.now()}, f)


def load_data_cached(ticker, days=1260):
    """Load OHLCV data with caching."""
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    cached_df, last_date = _load_cached_ohlcv(ticker)
    today = pd.Timestamp(datetime.today().date())

    if cached_df is not None and last_date >= today - pd.Timedelta(days=1):
        return cached_df

    if cached_df is not None:
        start = (last_date - timedelta(days=5)).strftime("%Y-%m-%d")
        end = datetime.today().strftime("%Y-%m-%d")
        try:
            raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            if not raw.empty:
                new_df = pd.DataFrame(index=raw.index)
                new_df["close"] = raw["Close"].values
                new_df["high"] = raw["High"].values
                new_df["low"] = raw["Low"].values
                new_df["open"] = raw["Open"].values
                new_df["volume"] = raw["Volume"].values.astype(float)
                vol_ticker = VOL_INDEX_MAP.get(ticker, None)
                if vol_ticker:
                    vol_series = _load_vol_index(vol_ticker,
                        pd.Timestamp(start), pd.Timestamp(end))
                    if vol_series is not None:
                        new_df["vix"] = vol_series.reindex(new_df.index, method="ffill").values
                    else:
                        vix_s = _load_vix_series(pd.Timestamp(start), pd.Timestamp(end))
                        new_df["vix"] = _compute_blended_vol(new_df, vix_s)
                else:
                    vix_s = _load_vix_series(pd.Timestamp(start), pd.Timestamp(end))
                    new_df["vix"] = _compute_blended_vol(new_df, vix_s)
                new_df.dropna(subset=["close"], inplace=True)
                new_dates = new_df.index.difference(cached_df.index)
                if len(new_dates) > 0:
                    base_cols = ["close", "high", "low", "open", "volume", "vix"]
                    combined = pd.concat([cached_df[base_cols], new_df.loc[new_dates, base_cols]])
                    combined = combined.sort_index()
                    combined = combined[~combined.index.duplicated(keep="last")]
                    _save_ohlcv_cache(ticker, combined)
                    return combined
        except Exception:
            pass
        return cached_df

    df = load_data(ticker, days=days)
    if df is not None and len(df) >= 300:
        _save_ohlcv_cache(ticker, df)
    return df


def process_ticker(ticker, days=1260, use_cache=True):
    """Process a single ticker end-to-end."""
    if use_cache:
        df = load_data_cached(ticker, days=days)
    else:
        df = load_data(ticker, days=days)

    if df is None or len(df) < 400:
        return None

    df = build_features(df)
    df = compute_trade_signal(df)
    df = classify_trade(df)
    history = compute_positions(df, compute_levels_last_n=1)
    return history


# =========================================================================
# MAIN
# =========================================================================

def run(tickers=None, days=1260, output="Equity_trend_positions.xlsx", use_cache=True):
    if tickers is None:
        tickers = DEFAULT_TICKERS

    cache_status = "ON" if use_cache else "OFF"
    print("=" * 70)
    print("  TREND + TRADE SIGNAL SYSTEM (Long-Only) — VIX Gate Enabled")
    print(f"  Tickers: {len(tickers)} | RF Window: {days}d | Cache: {cache_status}")
    print("=" * 70)

    rows = []
    failed = []
    for i, ticker in enumerate(tickers):
        print(f"\n  [{i+1}/{len(tickers)}] {ticker}...", end=" ", flush=True)
        try:
            history = process_ticker(ticker, days=days, use_cache=use_cache)
            if history is None or len(history) == 0:
                print("SKIP (insufficient data)")
                failed.append(ticker)
                continue
            row = build_output_row(ticker, history)
            rows.append(row)
            gate = row["VIX_Gate"]
            print(f"OK | {row['Position_Today']:>8} | {row['Action_Today']:<20} | "
                  f"Trend: {row['TREND_Regime']:<8} | Gate: {gate:<6} | "
                  f"Alloc: {row['Allocation']:.0%}")
        except Exception as e:
            print(f"ERROR: {e}")
            failed.append(ticker)
            continue

    if not rows:
        print("\nNo valid results.")
        return None

    rows.sort(key=lambda x: x["Ticker"])

    print(f"\n{'=' * 70}")
    print(f"  Writing output ({len(rows)} tickers)...")
    output_path = write_xlsx(rows, output)

    df_out = pd.DataFrame(rows)
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total tickers:  {len(rows)}")
    print(f"  Failed/skipped: {len(failed)} {failed if failed else ''}")
    print(f"  Hold Long:      {(df_out['Action_Today']=='Hold Long').sum()}")
    print(f"  Reduced:        {(df_out['Action_Today']=='Reduced Exposure').sum()}")
    print(f"  No Position:    {(df_out['Action_Today']=='No Position').sum()}")
    print(f"  Buy:            {(df_out['Action_Today']=='Buy').sum()}")
    print(f"  Increase Exp:   {(df_out['Action_Today']=='Increase Exposure').sum()}")
    print(f"  Sell:           {(df_out['Action_Today']=='Sell').sum()}")
    print(f"  Reduce Exp:     {(df_out['Action_Today']=='Reduce Exposure').sum()}")
    print(f"\n  Output: {output_path}")
    return df_out


# =========================================================================
# CLI + JUPYTER SUPPORT
# =========================================================================

TICKERS = DEFAULT_TICKERS
DAYS = 1260
OUTPUT = "Equity_trend_positions.xlsx"


def _is_notebook():
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except ImportError:
        pass
    return False


if __name__ == "__main__":
    if _is_notebook():
        run(tickers=TICKERS, days=DAYS, output=OUTPUT)
    else:
        import argparse
        parser = argparse.ArgumentParser(description="Trend + Trade Signal System (Long-Only) with VIX Gate")
        parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
        parser.add_argument("--days", type=int, default=1260)
        parser.add_argument("--output", default="Equity_trend_positions.xlsx")
        parser.add_argument("--no-cache", action="store_true", help="Disable caching")
        args = parser.parse_args()
        run(tickers=args.tickers, days=args.days, output=args.output, use_cache=not args.no_cache)
