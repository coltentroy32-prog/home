"""
Backtest: rf_blend_25 + VIX>EMA Trade Signal
=============================================
Replicates the 5-panel chart from the original system:
    1. Equity curve (System vs Buy & Hold)
    2. Target Allocation (0%, 50%, 100%)
    3. Drawdown (System vs B&H)
    4. System / B&H Ratio (>1 = outperforming)
    5. VIX Gate (VIX vs EMA63, colored by gate status)

Usage:
    python backtest.py                          # default: SPY, 2010-present
    python backtest.py --ticker QQQ --start 2012
    python backtest.py --ticker SPY --start 2010 --end 2024
    python backtest.py --no-plot --save results.csv

Requirements:
    pip install numpy pandas scipy scikit-learn yfinance matplotlib
"""
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from trend_system import (
    load_data, build_features, compute_trade_signal, classify_trade,
    compute_signals, compute_allocations,
)


def run_backtest(ticker="SPY", start_year=2010, end_year=None, days=3500):
    """Run backtest on a single ticker with the VIX-gated system.

    Args:
        ticker:     Ticker symbol to backtest
        start_year: Start year for the backtest period
        end_year:   End year (None = present)
        days:       Total trading days to load (needs enough for RF warmup)

    Returns:
        dict with backtest results, or None on failure
    """
    print(f"\n{'='*70}")
    print(f"  BACKTEST: {ticker} | {start_year} - {end_year or 'present'}")
    print(f"  System: rf_blend_25 + VIX>EMA Trade Signal")
    print(f"{'='*70}")

    # Load data — need enough history for RF warmup (1260d training + 200d features)
    end_date = datetime(end_year, 12, 31) if end_year else None
    print(f"\n  Loading {ticker} data...", end=" ", flush=True)
    df = load_data(ticker, days=days, end_date=end_date)
    if df is None or len(df) < 400:
        print(f"FAILED (insufficient data)")
        return None
    print(f"OK ({len(df)} bars, {df.index[0].date()} to {df.index[-1].date()})")

    # Build features and signals
    print(f"  Building features...", end=" ", flush=True)
    df = build_features(df)
    df = compute_trade_signal(df)
    df = classify_trade(df)
    print("OK")

    # Compute signal components
    print(f"  Computing RF + trend signals...", end=" ", flush=True)
    signals = compute_signals(df)
    print("OK")

    # Compute daily allocations (VIX-gated)
    allocation = compute_allocations(df, signals)

    # Daily returns
    close = df["close"].values
    daily_ret = np.zeros(len(df))
    daily_ret[1:] = close[1:] / close[:-1] - 1

    # System returns = allocation * daily return
    system_ret = allocation * daily_ret

    # Filter to backtest window
    start_date = pd.Timestamp(f"{start_year}-01-01")
    mask = df.index >= start_date
    if mask.sum() == 0:
        print(f"  ERROR: No data after {start_year}")
        return None

    dates = df.index[mask]
    sys_ret = pd.Series(system_ret[mask], index=dates)
    bh_ret = pd.Series(daily_ret[mask], index=dates)
    alloc_series = pd.Series(allocation[mask], index=dates)

    # Equity curves
    sys_equity = (1 + sys_ret).cumprod()
    bh_equity = (1 + bh_ret).cumprod()

    # Drawdowns
    sys_peak = sys_equity.cummax()
    sys_dd = (sys_equity - sys_peak) / sys_peak
    bh_peak = bh_equity.cummax()
    bh_dd = (bh_equity - bh_peak) / bh_peak

    # System / B&H ratio
    ratio = sys_equity / bh_equity

    # VIX data
    vix_vals = pd.Series(df["vix"].values[mask], index=dates)
    vix_ema = pd.Series(df["vix_ema63"].values[mask], index=dates)
    vix_gate_open = vix_vals > vix_ema

    # Signal states for annotation
    trend_st = pd.Series(signals["trend_state"][mask], index=dates)
    trade_st = pd.Series(signals["trade_state"][mask], index=dates)

    # Buy/Sell points (allocation transitions)
    alloc_arr = alloc_series.values
    buy_dates, buy_prices = [], []
    sell_dates, sell_prices = [], []
    close_bt = pd.Series(close[mask], index=dates)

    for i in range(1, len(alloc_arr)):
        if alloc_arr[i - 1] == 0 and alloc_arr[i] > 0:
            buy_dates.append(dates[i])
            buy_prices.append(close_bt.iloc[i])
        elif alloc_arr[i - 1] > 0 and alloc_arr[i] == 0:
            sell_dates.append(dates[i])
            sell_prices.append(close_bt.iloc[i])

    # Statistics
    trading_days = len(sys_ret)
    years = trading_days / 252

    sys_total_ret = sys_equity.iloc[-1] - 1
    bh_total_ret = bh_equity.iloc[-1] - 1

    sys_ann_ret = (1 + sys_total_ret) ** (1 / years) - 1
    bh_ann_ret = (1 + bh_total_ret) ** (1 / years) - 1

    sys_ann_vol = sys_ret.std() * np.sqrt(252)
    bh_ann_vol = bh_ret.std() * np.sqrt(252)

    sys_sharpe = sys_ann_ret / sys_ann_vol if sys_ann_vol > 0 else 0
    bh_sharpe = bh_ann_ret / bh_ann_vol if bh_ann_vol > 0 else 0

    sys_max_dd = sys_dd.min()
    bh_max_dd = bh_dd.min()

    # Time in market
    pct_invested = (alloc_series > 0).mean()
    pct_full = (alloc_series == 1.0).mean()
    pct_reduced = (alloc_series == 0.5).mean()
    pct_flat = (alloc_series == 0).mean()

    # Win rate of trades
    trades = []
    entry_price = None
    for i in range(1, len(alloc_arr)):
        if alloc_arr[i - 1] == 0 and alloc_arr[i] > 0:
            entry_price = close_bt.iloc[i]
        elif alloc_arr[i - 1] > 0 and alloc_arr[i] == 0 and entry_price is not None:
            trades.append(close_bt.iloc[i] / entry_price - 1)
            entry_price = None
    # Still in a trade at end
    if entry_price is not None and alloc_arr[-1] > 0:
        trades.append(close_bt.iloc[-1] / entry_price - 1)

    win_rate = np.mean([t > 0 for t in trades]) if trades else 0
    avg_win = np.mean([t for t in trades if t > 0]) if any(t > 0 for t in trades) else 0
    avg_loss = np.mean([t for t in trades if t <= 0]) if any(t <= 0 for t in trades) else 0

    stats = {
        "ticker": ticker,
        "period": f"{dates[0].date()} to {dates[-1].date()}",
        "years": round(years, 1),
        "system_return": sys_total_ret,
        "bh_return": bh_total_ret,
        "system_ann_return": sys_ann_ret,
        "bh_ann_return": bh_ann_ret,
        "system_sharpe": sys_sharpe,
        "bh_sharpe": bh_sharpe,
        "system_max_dd": sys_max_dd,
        "bh_max_dd": bh_max_dd,
        "pct_invested": pct_invested,
        "pct_full_position": pct_full,
        "pct_reduced": pct_reduced,
        "pct_flat": pct_flat,
        "num_trades": len(trades),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }

    # Print stats
    print(f"\n  {'─'*50}")
    print(f"  RESULTS: {ticker} ({dates[0].date()} to {dates[-1].date()})")
    print(f"  {'─'*50}")
    print(f"  System Return:    {sys_total_ret:>+8.1%}   |  B&H Return:   {bh_total_ret:>+8.1%}")
    print(f"  System Sharpe:    {sys_sharpe:>8.3f}   |  B&H Sharpe:   {bh_sharpe:>8.3f}")
    print(f"  System Max DD:    {sys_max_dd:>8.1%}   |  B&H Max DD:   {bh_max_dd:>8.1%}")
    print(f"  System Ann Ret:   {sys_ann_ret:>8.1%}   |  B&H Ann Ret:  {bh_ann_ret:>8.1%}")
    print(f"  {'─'*50}")
    print(f"  Time Invested:    {pct_invested:>8.1%}   (Full: {pct_full:.1%}, Reduced: {pct_reduced:.1%})")
    print(f"  Trades:           {len(trades):>8d}   |  Win Rate:     {win_rate:>8.1%}")
    if trades:
        print(f"  Avg Win:          {avg_win:>+8.1%}   |  Avg Loss:     {avg_loss:>+8.1%}")
    print(f"  {'─'*50}")

    result = {
        "stats": stats,
        "dates": dates,
        "sys_equity": sys_equity,
        "bh_equity": bh_equity,
        "allocation": alloc_series,
        "sys_dd": sys_dd,
        "bh_dd": bh_dd,
        "ratio": ratio,
        "vix": vix_vals,
        "vix_ema": vix_ema,
        "vix_gate_open": vix_gate_open,
        "buy_dates": buy_dates,
        "buy_prices": buy_prices,
        "sell_dates": sell_dates,
        "sell_prices": sell_prices,
        "close": close_bt,
    }

    return result


def plot_backtest(result, save_path=None):
    """Generate the 5-panel backtest chart matching the original system output.

    Panels:
        1. Equity curve with buy/sell markers
        2. Target allocation (0%, 50%, 100%)
        3. Drawdown comparison
        4. System / B&H performance ratio
        5. VIX Gate visualization
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.patches import Patch
    except ImportError:
        print("  ERROR: matplotlib not installed. Run: pip install matplotlib")
        return

    stats = result["stats"]
    dates = result["dates"]

    fig, axes = plt.subplots(5, 1, figsize=(16, 20), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1, 1.5, 1.5, 1.5]})

    fig.suptitle(
        f"FINAL SYSTEM: rf_blend_25 + VIX>EMA Trade Signal\n"
        f"Return: {stats['system_return']:+.1%} | Sharpe: {stats['system_sharpe']:.3f} | "
        f"MaxDD: {stats['system_max_dd']:.1%} | "
        f"vs B&H: {stats['bh_return']:+.1%}, Sharpe {stats['bh_sharpe']:.3f}, "
        f"DD {stats['bh_max_dd']:.1%}",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # ── Panel 1: Equity Curve ───────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(dates, result["sys_equity"], color="#1f77b4", linewidth=1.5,
             label=f"System ({stats['system_return']:+.1%})")
    ax1.plot(dates, result["bh_equity"], color="#aaaaaa", linewidth=1.0, alpha=0.7,
             label=f"Buy & Hold ({stats['bh_return']:+.1%})")

    # Buy/Sell markers
    if result["buy_dates"]:
        # Scale markers to equity curve
        buy_eq = result["sys_equity"].reindex(result["buy_dates"], method="nearest")
        ax1.scatter(result["buy_dates"], buy_eq * 0.97,
                    marker="^", color="#27ae60", s=40, zorder=5, alpha=0.8)
    if result["sell_dates"]:
        sell_eq = result["sys_equity"].reindex(result["sell_dates"], method="nearest")
        ax1.scatter(result["sell_dates"], sell_eq * 1.03,
                    marker="v", color="#e74c3c", s=40, zorder=5, alpha=0.8)

    ax1.set_ylabel("Growth of $1", fontsize=10)
    ax1.legend(loc="upper left", fontsize=9)
    ax1.set_title("Equity Curve", fontsize=11, fontweight="bold", loc="left")
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # ── Panel 2: Target Allocation ──────────────────────────────────────
    ax2 = axes[1]
    alloc = result["allocation"]

    # Color-fill allocation regions
    ax2.fill_between(dates, 0, alloc, where=(alloc == 1.0),
                     color="#27ae60", alpha=0.5, step="post", label="100%")
    ax2.fill_between(dates, 0, alloc, where=(alloc == 0.5),
                     color="#f39c12", alpha=0.5, step="post", label="50%")
    ax2.fill_between(dates, 0, alloc, where=(alloc == 0),
                     color="#e74c3c", alpha=0.2, step="post", label="0%")

    ax2.set_ylabel("Allocation", fontsize=10)
    ax2.set_ylim(-0.05, 1.15)
    ax2.set_yticks([0, 0.5, 1.0])
    ax2.set_yticklabels(["0%", "50%", "100%"])
    ax2.set_title("Target Allocation", fontsize=11, fontweight="bold", loc="left")
    ax2.legend(loc="upper right", fontsize=8, ncol=3)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Drawdown ───────────────────────────────────────────────
    ax3 = axes[2]
    ax3.fill_between(dates, result["sys_dd"], 0, color="#1f77b4", alpha=0.4,
                     label=f"System (max {stats['system_max_dd']:.1%})")
    ax3.fill_between(dates, result["bh_dd"], 0, color="#aaaaaa", alpha=0.3,
                     label=f"B&H (max {stats['bh_max_dd']:.1%})")
    ax3.set_ylabel("Drawdown", fontsize=10)
    ax3.set_title("Drawdown", fontsize=11, fontweight="bold", loc="left")
    ax3.legend(loc="lower left", fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    # ── Panel 4: System / B&H Ratio ────────────────────────────────────
    ax4 = axes[3]
    ax4.plot(dates, result["ratio"], color="#8e44ad", linewidth=1.2)
    ax4.axhline(1.0, color="#aaaaaa", linestyle="--", linewidth=0.8, alpha=0.7)
    ax4.fill_between(dates, 1.0, result["ratio"],
                     where=(result["ratio"] > 1), color="#27ae60", alpha=0.2)
    ax4.fill_between(dates, 1.0, result["ratio"],
                     where=(result["ratio"] < 1), color="#e74c3c", alpha=0.2)
    ax4.set_ylabel("Ratio", fontsize=10)
    ax4.set_title("System / B&H Ratio (>1 = outperforming)", fontsize=11,
                  fontweight="bold", loc="left")
    ax4.grid(True, alpha=0.3)

    # ── Panel 5: VIX Gate ───────────────────────────────────────────────
    ax5 = axes[4]
    vix = result["vix"]
    vix_ema = result["vix_ema"]
    gate_open = result["vix_gate_open"]

    ax5.plot(dates, vix, color="#e91e63", linewidth=0.8, alpha=0.8, label="VIX")
    ax5.plot(dates, vix_ema, color="#ff9800", linewidth=1.2, label="VIX EMA63")

    # Color background by gate status
    ax5.fill_between(dates, 0, vix.max() * 1.1,
                     where=gate_open, color="#ffcdd2", alpha=0.3, label="Gate OPEN (trade active)")
    ax5.fill_between(dates, 0, vix.max() * 1.1,
                     where=~gate_open, color="#c8e6c9", alpha=0.3, label="Gate CLOSED (fully invested)")

    ax5.set_ylabel("VIX", fontsize=10)
    ax5.set_title("VIX Gate: Trade cuts only when VIX > EMA63", fontsize=11,
                  fontweight="bold", loc="left")
    ax5.legend(loc="upper right", fontsize=8, ncol=2)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim(0, vix.max() * 1.15)

    # Format x-axis
    ax5.xaxis.set_major_locator(mdates.YearLocator())
    ax5.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=0, ha="center")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n  Chart saved: {save_path}")

    plt.show()
    return fig


def save_results_csv(result, path="backtest_results.csv"):
    """Save daily backtest data to CSV."""
    df = pd.DataFrame({
        "date": result["dates"],
        "close": result["close"].values,
        "system_equity": result["sys_equity"].values,
        "bh_equity": result["bh_equity"].values,
        "allocation": result["allocation"].values,
        "system_dd": result["sys_dd"].values,
        "bh_dd": result["bh_dd"].values,
        "ratio": result["ratio"].values,
        "vix": result["vix"].values,
        "vix_ema63": result["vix_ema"].values,
        "vix_gate_open": result["vix_gate_open"].values,
    })
    df.to_csv(path, index=False)
    print(f"  Results saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtest: rf_blend_25 + VIX>EMA Trade Signal System")
    parser.add_argument("--ticker", default="SPY", help="Ticker to backtest (default: SPY)")
    parser.add_argument("--start", type=int, default=2010, help="Start year (default: 2010)")
    parser.add_argument("--end", type=int, default=None, help="End year (default: present)")
    parser.add_argument("--days", type=int, default=3500,
                        help="Trading days to load for warmup (default: 3500)")
    parser.add_argument("--no-plot", action="store_true", help="Skip chart generation")
    parser.add_argument("--save-chart", default=None, help="Save chart to file (e.g. backtest.png)")
    parser.add_argument("--save-csv", default=None, help="Save daily results to CSV")
    args = parser.parse_args()

    result = run_backtest(
        ticker=args.ticker,
        start_year=args.start,
        end_year=args.end,
        days=args.days,
    )

    if result is None:
        sys.exit(1)

    if args.save_csv:
        save_results_csv(result, args.save_csv)

    if not args.no_plot:
        plot_backtest(result, save_path=args.save_chart)
