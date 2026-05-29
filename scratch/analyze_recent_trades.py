import pandas as pd
import numpy as np

def analyze():
    csv_path = r"c:\Personal Project\bot-crypto\logs\trades.csv"
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Ensure timestamp is string for easy sorting/filtering
    df['timestamp'] = df['timestamp'].astype(str)
    
    # Filter trades starting from 2026-05-21 18:15:41
    start_time = "2026-05-21 18:15:41"
    filtered_df = df[df['timestamp'] >= start_time].copy()
    
    if filtered_df.empty:
        print(f"No trades found starting from {start_time}")
        return
        
    total_trades = len(filtered_df)
    winning_trades = filtered_df[filtered_df['pnl_usdt'] > 0]
    losing_trades = filtered_df[filtered_df['pnl_usdt'] <= 0]
    
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    total_pnl_usdt = filtered_df['pnl_usdt'].sum()
    total_pnl_pct = filtered_df['pnl_pct'].sum()
    avg_pnl_pct = filtered_df['pnl_pct'].mean()

    output = []
    output.append("# RECENT TRADES ANALYSIS REPORT")
    output.append(f"Analyzing trades starting from: **{start_time}**\n")
    output.append("## Summary Statistics")
    output.append(f"- **Total Trades**: {total_trades}")
    output.append(f"- **Winning Trades**: {len(winning_trades)} ({win_rate*100:.2f}%)")
    output.append(f"- **Losing Trades**: {len(losing_trades)} ({(1-win_rate)*100:.2f}%)")
    output.append(f"- **Total PnL (USDT)**: {total_pnl_usdt:+.4f} USDT")
    output.append(f"- **Total PnL (%)**: {total_pnl_pct:+.2f}%")
    output.append(f"- **Avg PnL per Trade**: {avg_pnl_pct:+.4f}%\n")
    
    output.append("## Breakdown by Exit Reason")
    output.append("| Exit Reason | Count | Win Rate | Total PnL (USDT) | Total PnL (%) | Avg Duration |")
    output.append("|---|---|---|---|---|---|")
    exit_reasons = filtered_df['exit_reason'].value_counts()
    for reason, count in exit_reasons.items():
        reason_df = filtered_df[filtered_df['exit_reason'] == reason]
        reason_pnl_usdt = reason_df['pnl_usdt'].sum()
        reason_pnl_pct = reason_df['pnl_pct'].sum()
        avg_dur = reason_df['duration_min'].mean()
        win_count = len(reason_df[reason_df['pnl_usdt'] > 0])
        reason_win_rate = win_count / count if count > 0 else 0
        output.append(f"| {reason} | {count} | {reason_win_rate*100:.1f}% | {reason_pnl_usdt:+.4f} | {reason_pnl_pct:+.2f}% | {avg_dur:.1f}m |")

    output.append("\n## Breakdown by Symbol")
    output.append("| Symbol | Count | Win Rate | Total PnL (USDT) | Total PnL (%) | Avg Duration |")
    output.append("|---|---|---|---|---|---|")
    symbols = filtered_df['symbol'].value_counts()
    for sym, count in symbols.items():
        sym_df = filtered_df[filtered_df['symbol'] == sym]
        sym_pnl_usdt = sym_df['pnl_usdt'].sum()
        sym_pnl_pct = sym_df['pnl_pct'].sum()
        sym_win_rate = len(sym_df[sym_df['pnl_usdt'] > 0]) / count
        avg_dur = sym_df['duration_min'].mean()
        output.append(f"| {sym} | {count} | {sym_win_rate*100:.1f}% | {sym_pnl_usdt:+.4f} | {sym_pnl_pct:+.2f}% | {avg_dur:.1f}m |")

    output.append("\n## Breakdown by Side")
    output.append("| Side | Count | Win Rate | Total PnL (USDT) | Total PnL (%) | Avg Duration |")
    output.append("|---|---|---|---|---|---|")
    sides = filtered_df['side'].value_counts()
    for side, count in sides.items():
        side_df = filtered_df[filtered_df['side'] == side]
        side_pnl_usdt = side_df['pnl_usdt'].sum()
        side_pnl_pct = side_df['pnl_pct'].sum()
        side_win_rate = len(side_df[side_df['pnl_usdt'] > 0]) / count
        avg_dur = side_df['duration_min'].mean()
        output.append(f"| {side} | {count} | {side_win_rate*100:.1f}% | {side_pnl_usdt:+.4f} | {side_pnl_pct:+.2f}% | {avg_dur:.1f}m |")

    # Detailed view of losing trades
    output.append("\n## Detailed View of Unprofitable Trades")
    output.append("| Timestamp | Symbol | Side | PnL (USDT) | PnL (%) | Exit Reason | Duration |")
    output.append("|---|---|---|---|---|---|---|")
    losing_trades_sorted = losing_trades.sort_values(by='pnl_usdt')
    for idx, row in losing_trades_sorted.iterrows():
        output.append(f"| {row['timestamp']} | {row['symbol']} | {row['side']} | {row['pnl_usdt']:+.4f} | {row['pnl_pct']:+.2f}% | {row['exit_reason']} | {row['duration_min']:.1f}m |")

    report_path = r"c:\Personal Project\bot-crypto\scratch\recent_trades_analysis.md"
    with open(report_path, "w") as f:
        f.write("\n".join(output))
    print(f"Report written to {report_path}")

if __name__ == "__main__":
    analyze()
