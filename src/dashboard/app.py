"""
SCOPUS Vision Dashboard - INVESTISCOPE Trading Agent

Streamlit-based visualization dashboard for SCOPUS trading agent.
Displays equity curves, regime timeline, brain usage, and trade logs.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import sys
import os
from datetime import datetime, timedelta
import time
import numpy as np

# Add repo root to path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Import backtest engine
from src.backtest.engine import run_backtest, BacktestResult


def load_execution_logs(log_dir: str = "logs") -> pd.DataFrame:
    """Load execution logs from CSV files."""
    log_files = list(Path(log_dir).glob("*exec*.csv"))
    
    if not log_files:
        return pd.DataFrame()
    
    # Load most recent log
    latest_log = max(log_files, key=os.path.getctime)
    df = pd.read_csv(latest_log)
    
    return df


def load_session_metrics(summary_dir: str = "logs/summary") -> dict:
    """Load latest session metrics JSON."""
    summary_files = list(Path(summary_dir).glob("session_*.json"))
    
    if not summary_files:
        return {}
    
    # Load most recent summary
    latest_summary = max(summary_files, key=os.path.getctime)
    with open(latest_summary) as f:
        metrics = json.load(f)
    
    return metrics


def plot_equity_curve(df: pd.DataFrame):
    """Plot equity curve from trades."""
    if df.empty or 'pnl' not in df.columns:
        st.warning("No PnL data available")
        return
    
    # Calculate cumulative PnL
    df['cumulative_pnl'] = df['pnl'].cumsum()
    df['equity'] = 10000 + df['cumulative_pnl']  # Assume $10k starting capital
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['equity'],
        mode='lines',
        name='Equity',
        line=dict(color='#00ff00', width=2)
    ))
    
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Trade Number",
        yaxis_title="Equity ($)",
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig, width='stretch')


def plot_regime_timeline(df: pd.DataFrame):
    """Plot regime classification over time."""
    if df.empty or 'regime' not in df.columns:
        st.warning("No regime data available")
        return
    
    # Map regimes to colors
    regime_colors = {
        'TREND': '#00ff00',
        'RANGE': '#ffff00',
        'CRASH': '#ff0000',
        'UNCERTAIN': '#888888'
    }
    
    df['regime_color'] = df['regime'].map(regime_colors)
    
    fig = go.Figure()
    
    for regime in df['regime'].unique():
        regime_df = df[df['regime'] == regime]
        fig.add_trace(go.Scatter(
            x=regime_df.index,
            y=[regime] * len(regime_df),
            mode='markers',
            name=regime,
            marker=dict(
                color=regime_colors.get(regime, '#888888'),
                size=10
            )
        ))
    
    fig.update_layout(
        title="Regime Timeline",
        xaxis_title="Trade Number",
        yaxis_title="Regime",
        template="plotly_dark",
        height=300
    )
    
    st.plotly_chart(fig, width='stretch')


def plot_brain_usage(metrics: dict):
    """Plot brain usage pie chart."""
    if not metrics or 'brain_usage' not in metrics:
        st.warning("No brain usage data available")
        return
    
    brain_usage = metrics['brain_usage']
    
    # Filter out zero counts
    brain_usage = {k: v for k, v in brain_usage.items() if v > 0}
    
    if not brain_usage:
        st.warning("No brain usage data")
        return
    
    fig = go.Figure(data=[go.Pie(
        labels=list(brain_usage.keys()),
        values=list(brain_usage.values()),
        hole=0.3,
        marker=dict(colors=['#00ff00', '#0088ff', '#ff8800'])
    )])
    
    fig.update_layout(
        title="Brain Usage Distribution",
        template="plotly_dark",
        height=400
    )
    
    st.plotly_chart(fig, width='stretch')


def validate_and_adjust_date_range(
    price_data: pd.DataFrame,
    start_date: datetime,
    end_date: datetime
) -> tuple:
    """
    Validate date range against available data and auto-correct if needed.
    
    Returns:
        (adjusted_start, adjusted_end, warning_message)
    """
    if price_data.empty:
        return start_date, end_date, None
    
    # Get available date range from data
    data_start = price_data['timestamp'].min()
    data_end = price_data['timestamp'].max()
    
    # Convert to datetime if needed
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    adjusted_start = start_dt
    adjusted_end = end_dt
    warning = None
    
    # Check if requested range is outside available data
    if start_dt < data_start or end_dt > data_end:
        # Auto-correct to available range
        adjusted_start = max(start_dt, data_start)
        adjusted_end = min(end_dt, data_end)
        
        warning = (
            f"📅 Adjusted to available date range: "
            f"{adjusted_start.strftime('%d-%b-%Y')} → {adjusted_end.strftime('%d-%b-%Y')}"
        )
    
    return adjusted_start, adjusted_end, warning


def get_available_years(price_data: pd.DataFrame) -> list:
    """Extract available years from price data"""
    if price_data.empty:
        return []
    
    years = price_data['timestamp'].dt.year.unique()
    return sorted(years.tolist())


def prepare_price_data(price_data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare price data for charting.
    
    Ensures:
    - Timestamps are properly parsed and sorted
    - No duplicate timestamps
    - OHLC columns exist
    - Data is chronologically ordered
    """
    df = price_data.copy()
    
    # Parse timestamps with UTC
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    
    # Convert to tz-naive for easier comparison
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    
    # Drop rows with invalid timestamps
    df = df.dropna(subset=["timestamp"])
    
    # Sort chronologically (CRITICAL for proper candle spacing)
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # Remove duplicate timestamps (aggregate if needed)
    if df["timestamp"].duplicated().any():
        # Aggregate duplicates: take first open, max high, min low, last close
        df = df.groupby("timestamp", as_index=False).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum" if "volume" in df.columns else "first"
        })
    
    # Ensure required columns exist
    required_cols = ["open", "high", "low", "close"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    return df


def plot_tradingview_chart(
    symbol: str,
    price_data: pd.DataFrame,
    trades_df: pd.DataFrame = None,
    start_date: datetime = None,
    end_date: datetime = None,
    chart_key: str = "scopus_candles"
):
    """
    TradingView-style candlestick chart with clean, evenly-spaced candles.
    
    Features:
    - Properly ordered OHLC data (chronological)
    - No overlapping or squeezed candles
    - Smooth scroll zoom & pan
    - Trade overlays (entries, exits, SL/TP)
    - Dark TradingView theme
    """
    # 1. PREPARE DATA - Clean and order properly
    try:
        df = prepare_price_data(price_data)
    except Exception as e:
        st.error(f"Error preparing price data: {e}")
        return None, {}
    
    # 2. VALIDATE DATE RANGE
    if start_date and end_date:
        data_start = df["timestamp"].min()
        data_end = df["timestamp"].max()
        
        # Convert to datetime for comparison (ensure tz-naive)
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Remove timezone info for comparison (if present)
        if hasattr(data_start, 'tz') and data_start.tz is not None:
            data_start = data_start.tz_localize(None)
        if hasattr(data_end, 'tz') and data_end.tz is not None:
            data_end = data_end.tz_localize(None)
        if hasattr(start_dt, 'tz') and start_dt.tz is not None:
            start_dt = start_dt.tz_localize(None)
        if hasattr(end_dt, 'tz') and end_dt.tz is not None:
            end_dt = end_dt.tz_localize(None)
        
        # Check if requested range is outside available data
        if start_dt < data_start or end_dt > data_end:
            adjusted_start = max(start_dt, data_start)
            adjusted_end = min(end_dt, data_end)
            
            st.warning(
                f"📅 Selected range is partially outside available data: "
                f"data from **{data_start.strftime('%Y-%m-%d')}** to **{data_end.strftime('%Y-%m-%d')}**. "
                f"Backtest will be clipped to this range."
            )
            
            # Filter to adjusted range
            df = df[(df["timestamp"] >= adjusted_start) & (df["timestamp"] <= adjusted_end)]
        else:
            # Filter to requested range
            df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)]
    
    if df.empty:
        st.error("No data available for the selected date range")
        return None, {}
    
    # 3. CREATE CANDLESTICK CHART (TradingView Style)
    fig = go.Figure()
    
    # Add candlesticks with TradingView colors and THICK CANDLES
    fig.add_trace(go.Candlestick(
        x=df["timestamp"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name=symbol,
        increasing_line_color='#26a69a',  # TradingView green
        decreasing_line_color='#ef5350',  # TradingView red
        increasing_fillcolor='#26a69a',   # Solid fill (not transparent)
        decreasing_fillcolor='#ef5350',   # Solid fill (not transparent)
        line=dict(width=1),  # Thin wick lines
        increasing_line_width=1,
        decreasing_line_width=1,
        showlegend=False,
        # Make candles WIDER and more visible
        whiskerwidth=0.2,  # Thinner wicks relative to body
    ))
    
    # 4. OVERLAY TRADES (if provided)
    if trades_df is not None and not trades_df.empty:
        # Filter trades for this symbol
        symbol_trades = trades_df[trades_df['symbol'] == symbol].copy()
        
        if start_date and end_date:
            symbol_trades = symbol_trades[
                (pd.to_datetime(symbol_trades['timestamp_entry']) >= start_dt) &
                (pd.to_datetime(symbol_trades['timestamp_entry']) <= end_dt)
            ]
        
        if not symbol_trades.empty:
            # Separate buys and sells
            buys = symbol_trades[symbol_trades['side'] == 'buy']
            sells = symbol_trades[symbol_trades['side'] == 'sell']
            
            # Vectorized Buy Entries
            if not buys.empty:
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(buys['timestamp_entry']),
                    y=buys['entry_price'],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=12, color='#26a69a', line=dict(width=2, color='white')),
                    name='Buy Entry',
                    hovertemplate="<b>BUY ENTRY</b><br>Price: %{y:.5f}<br>Time: %{x}<extra></extra>"
                ))

            # Vectorized Sell Entries
            if not sells.empty:
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(sells['timestamp_entry']),
                    y=sells['entry_price'],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=12, color='#ef5350', line=dict(width=2, color='white')),
                    name='Sell Entry',
                    hovertemplate="<b>SELL ENTRY</b><br>Price: %{y:.5f}<br>Time: %{x}<extra></extra>"
                ))
            
            # Vectorized Exits
            exits = symbol_trades[symbol_trades['timestamp_exit'].notna()]
            if not exits.empty:
                # Color based on PnL
                colors = ['#26a69a' if pnl > 0 else '#ef5350' for pnl in exits['pnl']]
                
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(exits['timestamp_exit']),
                    y=exits['exit_price'],
                    mode='markers',
                    marker=dict(symbol='x', size=10, color=colors, line=dict(width=2, color='white')),
                    name='Exit',
                    text=[f"PnL: ${pnl:.2f}" for pnl in exits['pnl']],
                    hovertemplate="<b>EXIT</b><br>Price: %{y:.5f}<br>Time: %{x}<br>%{text}<extra></extra>"
                ))

            # Add SL/TP lines only if trade count is manageable
            if len(symbol_trades) <= 200:
                for _, trade in symbol_trades.iterrows():
                    entry_ts = pd.to_datetime(trade['timestamp_entry'])
                    if 'timestamp_exit' in trade and pd.notna(trade['timestamp_exit']):
                        exit_ts = pd.to_datetime(trade['timestamp_exit'])
                        
                        # SL Line
                        if 'sl_price' in trade and pd.notna(trade['sl_price']):
                            fig.add_shape(type='line', x0=entry_ts, x1=exit_ts, y0=trade['sl_price'], y1=trade['sl_price'],
                                        line=dict(color='rgba(239, 83, 80, 0.5)', width=1, dash='dash'), layer='below')
                        
                        # TP Line
                        if 'tp_price' in trade and pd.notna(trade['tp_price']):
                            fig.add_shape(type='line', x0=entry_ts, x1=exit_ts, y0=trade['tp_price'], y1=trade['tp_price'],
                                        line=dict(color='rgba(38, 166, 154, 0.5)', width=1, dash='dash'), layer='below')
            else:
                # Add a note about hidden lines
                fig.add_annotation(
                    text=f"⚠️ SL/TP lines hidden for performance ({len(symbol_trades)} trades)",
                    xref="paper", yref="paper",
                    x=0.01, y=0.99, showarrow=False,
                    font=dict(color="yellow", size=10),
                    bgcolor="rgba(0,0,0,0.5)"
                )
    
    # 5. TRADINGVIEW-STYLE LAYOUT
    fig.update_layout(
        title=dict(
            text=f"{symbol} – Candlestick Chart",
            font=dict(size=18, color='white')
        ),
        xaxis=dict(
            title="Time",
            gridcolor='#1e1e1e',
            showgrid=True,
            rangeslider=dict(visible=False),  # NO range slider for clean look
            type='date',
            # Remove only weekend gaps for cleaner look
            rangebreaks=[
                dict(bounds=["sat", "mon"])  # Hide weekends only
            ],
            # Better tick spacing for clearer candles
            tickmode='auto',
            nticks=20,  # Limit number of ticks for cleaner look
            # Ensure proper spacing
            automargin=True
        ),
        yaxis=dict(
            title="Price",
            gridcolor='#1e1e1e',
            showgrid=True,
            side='right'  # TradingView-style: price on right
        ),
        template='plotly_dark',
        hovermode='x unified',
        height=650,
        plot_bgcolor='#131722',  # TradingView dark background
        paper_bgcolor='#131722',
        font=dict(color='#d1d4dc'),
        dragmode='pan',  # Default to PAN (not zoom)
        margin=dict(l=40, r=80, t=60, b=40),
        # Smooth transitions
        transition=dict(duration=200, easing='cubic-in-out'),
        # Persistent zoom state
        uirevision=chart_key
    )
    
    # 6. CONFIGURE SMOOTH INTERACTIONS
    config = {
        'scrollZoom': True,  # Mouse wheel = zoom
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['autoScale2d', 'lasso2d', 'select2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': f'scopus_{symbol}_chart',
            'height': 1080,
            'width': 1920,
            'scale': 2
        }
    }
    
    return fig, config


def main():
    st.set_page_config(
        page_title="INVESTISCOPE – SCOPUS Trading Dashboard",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 INVESTISCOPE – SCOPUS Trading Dashboard")
    st.markdown("Real-time trading agent visualization powered by SCOPUS")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📊 Live Results", "🔬 Backtesting"])
    
    # ========== TAB 1: LIVE RESULTS ==========
    with tab1:
        # Sidebar controls
        st.sidebar.header("Controls")
        log_dir = st.sidebar.text_input("Log Directory", value="logs")
        summary_dir = st.sidebar.text_input("Summary Directory", value="logs/summary")
        
        if st.sidebar.button("Refresh Data"):
            st.rerun()
        
        # Load data
        df = load_execution_logs(log_dir)
        metrics = load_session_metrics(summary_dir)
        
        # Metrics row
        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Trades", metrics.get('total_trades', 0))
            
            with col2:
                winrate = metrics.get('winrate', 0) * 100
                st.metric("Winrate", f"{winrate:.1f}%")
            
            with col3:
                total_return = metrics.get('total_return', 0) * 100
                st.metric("Total Return", f"{total_return:.2f}%")
            
            with col4:
                sharpe = metrics.get('sharpe_ratio', 0)
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Equity Curve")
            plot_equity_curve(df)
        
        with col2:
            st.subheader("🧠 Brain Usage")
            plot_brain_usage(metrics)
        
        # Regime timeline
        st.subheader("🌐 Regime Timeline")
        plot_regime_timeline(df)
        
        # Trade log table
        st.subheader("📋 Trade Log")
        
        if not df.empty:
            # Add filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'symbol' in df.columns:
                    symbols = ['All'] + list(df['symbol'].unique())
                    selected_symbol = st.selectbox("Symbol", symbols)
            
            with col2:
                if 'decision_source' in df.columns:
                    sources = ['All'] + list(df['decision_source'].unique())
                    selected_source = st.selectbox("Decision Source", sources)
            
            with col3:
                if 'regime' in df.columns:
                    regimes = ['All'] + list(df['regime'].unique())
                    selected_regime = st.selectbox("Regime", regimes)
            
            # Apply filters
            filtered_df = df.copy()
            
            if 'symbol' in df.columns and selected_symbol != 'All':
                filtered_df = filtered_df[filtered_df['symbol'] == selected_symbol]
            
            if 'decision_source' in df.columns and selected_source != 'All':
                filtered_df = filtered_df[filtered_df['decision_source'] == selected_source]
            
            if 'regime' in df.columns and selected_regime != 'All':
                filtered_df = filtered_df[filtered_df['regime'] == selected_regime]
            
            # Display table
            st.dataframe(
                filtered_df,
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="jarvis_trades.csv",
                mime="text/csv"
            )
        else:
            st.info("No trade data available. Run executor to generate logs.")
    
    # ========== TAB 2: BACKTESTING ==========
    with tab2:
        st.header("🔬 Interactive Backtesting")
        st.markdown("Run backtests on historical CSV data with custom date ranges")
        
        # Backtesting controls
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📅 Date Range")
            
            # Detect available data range from CSV files
            csv_files = list(Path("data/raw/forex_kaggle_multiTF").glob("*_M15.csv"))
            
            if csv_files:
                # Load first CSV to get date range
                sample_df = pd.read_csv(csv_files[0])
                sample_df['timestamp'] = pd.to_datetime(sample_df['timestamp'])
                
                data_min_date = sample_df['timestamp'].min().date()
                data_max_date = sample_df['timestamp'].max().date()
                
                # Show available data range with prominent warning
                st.info(
                    f"📊 **Available Data Range:**\n\n"
                    f"From: **{data_min_date.strftime('%d-%b-%Y')}** ({data_min_date.year})\n\n"
                    f"To: **{data_max_date.strftime('%d-%b-%Y')}** ({data_max_date.year})"
                )
                
                st.warning(
                    f"⚠️ **Important:** When using the date picker, make sure to select year "
                    f"**{data_min_date.year}** or **{data_max_date.year}** from the year dropdown. "
                    f"Other years have no data!"
                )
                
                # Add note for Custom mode
                st.info(
                    "💡 **Tip:** In Custom mode, you can select dates from **both 2024 and 2025**. "
                    "Click the **year dropdown** in the date picker and select **2024** to choose 2024 dates!"
                )
                
                # Quick duration selector
                st.markdown("**⚡ Quick Duration Selection**")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    # Year toggle
                    year_choice = st.radio(
                        "Select Year Range",
                        ["2024 Data (Sep-Dec)", "2025 Data (Jan-Aug)", "Custom"],
                        index=0,
                        horizontal=False
                    )
                
                with col_b:
                    # Duration buttons - only show if NOT Custom
                    if year_choice != "Custom":
                        duration_choice = st.radio(
                            "Select Duration",
                            ["1 Year", "6 Months", "3 Months", "1 Month", "10 Days"],
                            index=2,  # Default to 3 months
                            horizontal=False
                        )
                    else:
                        st.markdown("**Manual Selection**")
                        st.info("📅 Manually select your exact start and end dates below")
                        duration_choice = None  # No preset duration for Custom
                
                # Calculate dates based on selections
                if year_choice == "2024 Data (Sep-Dec)":
                    # 2024 range: Sep 1 - Dec 31, 2024
                    range_start = data_min_date  # Sep 1, 2024
                    range_end = datetime(2024, 12, 31).date()
                elif year_choice == "2025 Data (Jan-Aug)":
                    # 2025 range: Jan 1 - Aug 31, 2025
                    range_start = datetime(2025, 1, 1).date()
                    range_end = data_max_date  # Aug 31, 2025
                else:
                    # Custom - allow FULL range (both 2024 & 2025)
                    range_start = data_min_date  # Sep 1, 2024
                    range_end = data_max_date    # Aug 31, 2025
                
                # Apply duration (only for non-Custom modes)
                if year_choice == "Custom":
                    # Custom mode: default to full range, user will manually adjust
                    default_start = data_min_date  # Sep 1, 2024
                    default_end = data_max_date    # Aug 31, 2025
                elif duration_choice == "1 Year":
                    default_start = range_start
                    default_end = min(range_end, range_start + timedelta(days=365))
                elif duration_choice == "6 Months":
                    default_start = range_start
                    default_end = min(range_end, range_start + timedelta(days=180))
                elif duration_choice == "3 Months":
                    default_start = range_start
                    default_end = min(range_end, range_start + timedelta(days=90))
                elif duration_choice == "1 Month":
                    default_start = range_start
                    default_end = min(range_end, range_start + timedelta(days=30))
                else:  # 10 Days
                    default_start = range_start
                    default_end = min(range_end, range_start + timedelta(days=10))
            else:
                # Fallback if no CSV files
                st.warning("⚠️ No price data found in data/raw/forex_kaggle_multiTF folder")
                data_min_date = datetime(2024, 1, 1).date()
                data_max_date = datetime.now().date()
                default_start = data_max_date - timedelta(days=90)
                default_end = data_max_date
            
            start_date = st.date_input(
                "Start Date",
                value=default_start,
                min_value=data_min_date,
                max_value=data_max_date,
                help=f"⚠️ IMPORTANT: Select year {data_min_date.year} or {data_max_date.year} only!"
            )
            
            end_date = st.date_input(
                "End Date",
                value=default_end,
                min_value=data_min_date,
                max_value=data_max_date,
                help=f"⚠️ IMPORTANT: Select year {data_min_date.year} or {data_max_date.year} only!"
            )
            
            # Validate date range
            date_error = False
            
            if start_date > end_date:
                st.error("❌ **Start date must be before end date!**")
                date_error = True
            elif start_date < data_min_date or end_date > data_max_date:
                st.error(
                    f"❌ **No data available for selected range!**\n\n"
                    f"Available data: **{data_min_date.strftime('%d-%b-%Y')}** to **{data_max_date.strftime('%d-%b-%Y')}**\n\n"
                    f"You selected: **{start_date.strftime('%d-%b-%Y')}** to **{end_date.strftime('%d-%b-%Y')}**"
                )
                date_error = True
            elif start_date.year < data_min_date.year or end_date.year > data_max_date.year:
                st.error(
                    f"❌ **Invalid year selected!**\n\n"
                    f"Data available only for years: **{data_min_date.year}** to **{data_max_date.year}**\n\n"
                    f"Please select dates within the available range."
                )
                date_error = True
            else:
                days_selected = (end_date - start_date).days
                st.success(
                    f"✅ **Valid range selected!**\n\n"
                    f"From: **{start_date.strftime('%d-%b-%Y')}**\n\n"
                    f"To: **{end_date.strftime('%d-%b-%Y')}**\n\n"
                    f"Duration: **{days_selected} days**"
                )
        
        with col2:
            st.subheader("⚙️ Configuration")
            
            # Symbol selection - dynamically extract from available data
            import glob
            data_files = glob.glob("data/raw/forex_kaggle_multiTF/*_M15.csv")
            available_symbols = sorted(list(set([
                os.path.basename(f).split('_')[0] for f in data_files
            ])))
            
            if not available_symbols:
                st.error("❌ No M15 data found in data/raw/forex_kaggle_multiTF/")
                available_symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']  # Fallback
            
            selected_symbols = st.multiselect(
                "Symbols",
                available_symbols,
                default=[available_symbols[0]] if available_symbols else []
            )
            
            # Timeframe selection
            st.markdown("**Timeframe**")
            timeframe_options = {
                '5 Minutes (M5)': 'M5',
                '15 Minutes (M15)': 'M15',
                '30 Minutes (M30)': 'M30',
                '1 Hour (H1)': 'H1',
                '4 Hours (H4)': 'H4',
                '1 Day (D1)': 'D1',
                '1 Week (W1)': 'W1',
                '1 Month (MN1)': 'MN1'
            }
            selected_timeframe_label = st.selectbox(
                "Select Timeframe",
                list(timeframe_options.keys()),
                index=1  # Default to M15
            )
            selected_timeframe = timeframe_options[selected_timeframe_label]
            
            # SCOPUS features
            st.markdown("**SCOPUS Features**")
            enable_meta_gating = st.checkbox("Meta-Gating Brain", value=False)
            enable_portfolio = st.checkbox("Portfolio Brain", value=False)
            enable_slicer = st.checkbox("Execution Slicer", value=False)
            
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=1000,
                max_value=1000000,
                value=10000,
                step=1000
            )
        
        # Run backtest button
        st.markdown("---")
        
        # Disable button if date range is invalid
        button_disabled = date_error if 'date_error' in locals() else False
        
        if button_disabled:
            st.warning("⚠️ Please fix the date range errors above before running backtest")
        
        if st.button("🚀 Run Backtest", type="primary", use_container_width=True, disabled=button_disabled):
            if not selected_symbols:
                st.error("❌ Please select at least one symbol")
            else:
                with st.spinner("Running backtest... ⏳"):
                    try:
                        # Convert dates to datetime
                        start_dt = datetime.combine(start_date, datetime.min.time())
                        end_dt = datetime.combine(end_date, datetime.max.time())
                        
                        # Run backtest with REAL 13-stage pipeline
                        result = run_backtest(
                            symbols=selected_symbols,
                            start_date=start_dt,
                            end_date=end_dt,
                            initial_capital=initial_capital,
                            enable_meta_gating=enable_meta_gating,
                            enable_portfolio_brain=enable_portfolio,
                            enable_slicer=enable_slicer,
                            enable_rl_fallback=False,
                            csv_price_dir="data/raw/forex_kaggle_multiTF",
                            output_dir="logs/backtest",
                            primary_model_path="models/fx_bin_19f_thresh55__pack_ok.joblib",
                            finrl_policies_path="models/finrl/",
                            use_real_pipeline=True
                        )
                        
                        # Store result and dates in session state
                        st.session_state['backtest_result'] = result
                        st.session_state['backtest_start_date'] = start_date
                        st.session_state['backtest_end_date'] = end_date
                        
                        st.success(f"✅ Backtest completed in {result.execution_time:.2f}s!")
                    
                    except Exception as e:
                        st.error(f"❌ Backtest failed: {str(e)}")
                        st.exception(e)
        
        # Display results
        if 'backtest_result' in st.session_state:
            result = st.session_state['backtest_result']
            
            # Get dates from session state
            start_date = st.session_state.get('backtest_start_date', datetime.now() - timedelta(days=90))
            end_date = st.session_state.get('backtest_end_date', datetime.now())
            
            st.markdown("---")
            st.header("📊 Backtest Results")
            
            # Metrics row
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("💰 Total Return", f"{result.total_return*100:.2f}%", f"${result.total_pnl:,.2f}")
            with col2:
                st.metric("🎯 Win Rate", f"{result.winrate*100:.1f}%", f"{result.winning_trades}/{result.total_trades}")
            with col3:
                st.metric("📉 Max Drawdown", f"{result.max_drawdown*100:.2f}%")
            with col4:
                st.metric("⚖️ Profit Factor", f"{result.profit_factor:.2f}")
            with col5:
                st.metric("📊 Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
            
            # Secondary metrics (Gating & Activity)
            st.markdown("")
            m1, m2, m3, m4, m5 = st.columns(5)
            
            # Calculate these on the fly if not in result object
            days = (pd.to_datetime(result.end_date) - pd.to_datetime(result.start_date)).days
            avg_trades_day = result.total_trades / days if days > 0 else 0
            trades_per_100 = (result.total_trades / result.total_candles * 100) if hasattr(result, 'total_candles') and result.total_candles > 0 else 0
            
            with m1:
                st.metric("⚡ Avg Trades/Day", f"{avg_trades_day:.1f}")
            with m2:
                st.metric("🕯️ Trades/100 Bars", f"{trades_per_100:.2f}")
            with m3:
                st.metric("⏱️ Avg Duration", f"{result.avg_trade_duration_minutes:.0f} min")
            with m4:
                st.metric("🎲 Exposure", f"{result.exposure_pct*100:.1f}%")
            with m5:
                st.metric("🔄 Total Trades", f"{result.total_trades}")

            st.markdown("---")
            # Use columns to align selector and info
            c1, c2 = st.columns([1, 3])
            with c1:
                selected_symbol = st.selectbox(
                    "📊 Select Symbol",
                    options=result.symbols,
                    index=0,
                    key="chart_symbol_selector_unique"
                )
            
            # Load price data for selected symbol - Kaggle format is simpler: SYMBOL_M15.csv
            price_csv = f"data/raw/forex_kaggle_multiTF/{selected_symbol}_M15.csv"
            
            if price_csv and os.path.exists(price_csv):
                price_data = pd.read_csv(price_csv)
                price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
                
                # Show available years
                available_years = get_available_years(price_data)
                with c2:
                    if available_years:
                        st.info(f"📅 Data Available: {', '.join(map(str, available_years))}")
                
                # TradingView-style candlestick chart
                st.subheader(f"📈 {selected_symbol} - TradingView Chart")
                
                # Convert trades to DataFrame
                trades_df = pd.DataFrame(result.trades) if result.trades else pd.DataFrame()
                
                # Plot TradingView chart
                try:
                    chart, config = plot_tradingview_chart(
                        symbol=selected_symbol,
                        price_data=price_data,
                        trades_df=trades_df if not trades_df.empty else None,
                        start_date=start_date,
                        end_date=end_date,
                        chart_key=f"scopus_{selected_symbol}_{int(datetime.now().timestamp())}"
                    )
                    
                    if chart is not None:
                        # Force height and width
                        chart.update_layout(height=700)
                        st.plotly_chart(chart, use_container_width=True, config=config, key=f"chart_{selected_symbol}")
                        
                        # Chart info
                        if not trades_df.empty:
                            symbol_trades = trades_df[trades_df['symbol'] == selected_symbol]
                            st.success(f"📍 Displaying {len(symbol_trades)} trades for {selected_symbol}")
                        else:
                            st.warning("⚠️ No trades to display for this symbol in the selected period.")
                    else:
                        st.error("❌ Failed to generate chart (data might be empty for range)")
                except Exception as e:
                    st.error(f"❌ Chart Error: {str(e)}")
                    st.exception(e)
            else:
                st.error(f"❌ Price data not found: {price_csv}")
            
            # Equity curve
            st.markdown("---")
            st.subheader("💰 Equity Curve")
            
            # Convert equity_curve to proper format
            if result.equity_curve:
                equity_df = pd.DataFrame(result.equity_curve)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(equity_df['timestamp']),
                    y=equity_df['equity'],
                    mode='lines',
                    name='Equity',
                    line=dict(color='#26a69a', width=2),
                    fill='tozeroy',
                    fillcolor='rgba(38, 166, 154, 0.1)'
                ))
                
                fig.update_layout(
                    xaxis_title="Time",
                    yaxis_title="Equity ($)",
                    template="plotly_dark",
                    height=300,
                    plot_bgcolor='#131722',
                    paper_bgcolor='#131722'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Fallback for old format
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=result.equity_curve,
                    mode='lines',
                    name='Equity',
                    line=dict(color='#00ff00', width=2)
                ))
                
                fig.update_layout(
                    xaxis_title="Trade Number",
                    yaxis_title="Equity ($)",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, width='stretch')
            
            # Brain usage and FinRL analysis
            st.markdown("---")
            st.header("🧠 Decision Source Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Brain Usage Distribution")
                
                # Use decision_source_breakdown instead of brain_usage
                if hasattr(result, 'decision_source_breakdown') and result.decision_source_breakdown:
                    brain_data = {k: v['trades'] for k, v in result.decision_source_breakdown.items() if v['trades'] > 0}
                    
                    if brain_data:
                        # Create pie chart with custom colors
                        colors = {
                            'PRIMARY': '#26a69a',      # Green - main model
                            'RL_FALLBACK': '#0088ff',  # Blue - FinRL
                            'HEURISTIC': '#ff8800'     # Orange - fallback
                        }
                        chart_colors = [colors.get(k, '#888888') for k in brain_data.keys()]
                        
                        fig = go.Figure(data=[go.Pie(
                            labels=list(brain_data.keys()),
                            values=list(brain_data.values()),
                            hole=0.4,
                            marker=dict(colors=chart_colors),
                            textinfo='label+percent',
                            textfont=dict(size=14, color='white')
                        )])
                        
                        fig.update_layout(
                            template="plotly_dark",
                            height=350,
                            plot_bgcolor='#131722',
                            paper_bgcolor='#131722',
                            showlegend=True,
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=-0.2,
                                xanchor="center",
                                x=0.5
                            )
                        )
                        
                        st.plotly_chart(fig, width='stretch')
                        
                        # Show total trades
                        total_trades = sum(brain_data.values())
                        st.metric("Total Trades", total_trades)
                else:
                    st.info("No brain usage data available")
            
            with col2:
                st.subheader("🤖 FinRL Performance Metrics")
                
                if hasattr(result, 'decision_source_breakdown') and result.decision_source_breakdown:
                    # Extract FinRL metrics
                    finrl_data = result.decision_source_breakdown.get('RL_FALLBACK', {})
                    primary_data = result.decision_source_breakdown.get('PRIMARY', {})
                    
                    if finrl_data and finrl_data.get('trades', 0) > 0:
                        # FinRL metrics
                        finrl_trades = finrl_data['trades']
                        finrl_winrate = finrl_data['winrate'] * 100
                        finrl_avg_pnl = finrl_data['avg_pnl']
                        
                        # Display FinRL stats
                        st.markdown("### 🔵 RL_FALLBACK (FinRL)")
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Trades", finrl_trades)
                        with col_b:
                            st.metric("Winrate", f"{finrl_winrate:.1f}%")
                        with col_c:
                            st.metric("Avg P&L", f"${finrl_avg_pnl:.2f}")
                        
                        # Compare with PRIMARY
                        if primary_data and primary_data.get('trades', 0) > 0:
                            st.markdown("### 🟢 PRIMARY vs 🔵 FinRL")
                            
                            primary_winrate = primary_data['winrate'] * 100
                            primary_avg_pnl = primary_data['avg_pnl']
                            
                            # Comparison metrics
                            wr_diff = finrl_winrate - primary_winrate
                            pnl_diff = finrl_avg_pnl - primary_avg_pnl
                            
                            col_x, col_y = st.columns(2)
                            with col_x:
                                st.metric(
                                    "Winrate Diff",
                                    f"{wr_diff:+.1f}%",
                                    delta=f"{wr_diff:+.1f}%",
                                    delta_color="normal"
                                )
                            with col_y:
                                st.metric(
                                    "P&L Diff",
                                    f"${pnl_diff:+.2f}",
                                    delta=f"${pnl_diff:+.2f}",
                                    delta_color="normal"
                                )
                            
                            # Performance indicator
                            if finrl_winrate >= primary_winrate:
                                st.success("✅ FinRL performing well as fallback!")
                            else:
                                st.warning("⚠️ FinRL underperforming PRIMARY (expected for grey-zone trades)")
                    else:
                        st.info("ℹ️ No FinRL trades in this backtest. FinRL activates only in grey-zone (0.40-0.70 confidence).")
                        
                        # Show explanation
                        msg = (
                            "**FinRL Activation Conditions:**\n"
                            "- PRIMARY confidence: 0.40 - 0.70 (grey zone)\n"
                            "- FinRL confidence: >= 0.65\n"
                            "- Position size: 40% (reduced risk)"
                        )
                        st.markdown(msg)
                else:
                    st.info("No decision source data available")
            
            # Regime distribution
            st.markdown("---")
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("🌐 Regime Distribution")
                
                # Use regime_breakdown instead of regime_counts
                if hasattr(result, 'regime_breakdown') and result.regime_breakdown:
                    regime_data = {k: v['trades'] for k, v in result.regime_breakdown.items()}
                    
                    fig = go.Figure(data=[go.Bar(
                        x=list(regime_data.keys()),
                        y=list(regime_data.values()),
                        marker=dict(color='#26a69a')
                    )])
                    
                    fig.update_layout(
                        template="plotly_dark",
                        height=300,
                        xaxis_title="Regime",
                        yaxis_title="Count",
                        plot_bgcolor='#131722',
                        paper_bgcolor='#131722'
                    )
                    
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.info("No regime data available")
            
            # Trade log
            st.subheader("📋 Trade Log")
            
            if result.trades:
                trades_df = pd.DataFrame(result.trades)
                st.dataframe(trades_df, use_container_width=True, height=400)
                
                # Download button
                csv = trades_df.to_csv(index=False)
                st.download_button(
                    label="Download Backtest Results",
                    data=csv,
                    file_name=f"backtest_{start_date}_{end_date}.csv",
                    mime="text/csv"
                )
    
    # Footer
    st.markdown("---")
    st.markdown("**INVESTISCOPE** | SCOPUS Trading Agent | Powered by Advanced AI 🚀")


if __name__ == "__main__":
    main()
