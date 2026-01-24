"""
SCOPUS Jarvis HUD - Backend API Server

Provides REST API endpoints for:
- Running backtests
- Getting agent status
- Real-time market data
- Trade execution

Run with: python src/dashboard/jarvis_api.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import subprocess
import json
import os
from pathlib import Path
from datetime import datetime
import threading

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Global state
backtest_status = {
    'running': False,
    'progress': 0,
    'current_symbol': None,
    'results': None
}

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/agent/status', methods=['GET'])
def get_agent_status():
    """Get current agent status"""
    return jsonify({
        'ml_confidence': 78,
        'regime': 'RANGE',
        'risk_mode': 'NORMAL',
        'rl_state': 'EXPLORING',
        'active_strategy': 'ML_BRAIN_V1',
        'diagnostics': {
            'feature_health': 92,
            'strategy_activation': 78,
            'signal_noise': 0.15,
            'model_drift': 0.03
        }
    })

@app.route('/api/market/overview', methods=['GET'])
def get_market_overview():
    """Get market overview data"""
    symbol = request.args.get('symbol', 'EURUSD')
    timeframe = request.args.get('timeframe', 'M15')
    
    return jsonify({
        'symbol': symbol,
        'timeframe': timeframe,
        'metrics': {
            'trend_strength': 65,
            'volatility': 42,
            'liquidity': 88
        },
        'positions': 3,
        'pnl_today': 247.50
    })

@app.route('/api/backtest/run', methods=['POST'])
def run_backtest():
    """
    Run backtest with specified parameters
    
    Request body:
    {
        "symbol": "EURUSD",
        "start_date": "2023-01-01",
        "end_date": "2023-01-07",
        "initial_capital": 10000
    }
    """
    global backtest_status
    
    if backtest_status['running']:
        return jsonify({'error': 'Backtest already running'}), 400
    
    data = request.json
    symbol = data.get('symbol', 'EURUSD')
    start_date = data.get('start_date', '2023-01-01')
    end_date = data.get('end_date', '2023-01-07')
    initial_capital = data.get('initial_capital', 10000)
    
    # Update status
    backtest_status['running'] = True
    backtest_status['progress'] = 0
    backtest_status['current_symbol'] = symbol
    backtest_status['results'] = None
    
    # Run backtest in background thread
    thread = threading.Thread(
        target=execute_backtest,
        args=(symbol, start_date, end_date, initial_capital)
    )
    thread.start()
    
    return jsonify({
        'status': 'started',
        'symbol': symbol,
        'start_date': start_date,
        'end_date': end_date
    })

@app.route('/api/backtest/status', methods=['GET'])
def get_backtest_status():
    """Get current backtest status and progress"""
    return jsonify(backtest_status)

@app.route('/api/backtest/results', methods=['GET'])
def get_backtest_results():
    """Get latest backtest results"""
    results_dir = Path('backtest_results')
    
    if not results_dir.exists():
        return jsonify({'error': 'No results found'}), 404
    
    # Read summary.json
    summary_file = results_dir / 'summary.json'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            results = json.load(f)
        return jsonify(results)
    else:
        return jsonify({'error': 'Results not available'}), 404

def execute_backtest(symbol, start_date, end_date, initial_capital):
    """Execute backtest with simulated smooth progress"""
    global backtest_status
    
    try:
        import time
        
        # Build command
        venv_python = Path('.venv311/Scripts/python.exe')
        if not venv_python.exists():
            venv_python = 'python'
        
        cmd = [
            str(venv_python), '-m', 'src.backtest.engine',
            '--config', 'config/mvp_v1.yaml',
            '--symbols', symbol,
            '--start', start_date,
            '--end', end_date
        ]
        
        print(f"Running backtest: {' '.join(cmd)}")
        print(f"Working directory: {os.getcwd()}")
        
        # Start subprocess
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.getcwd()
        )
        
        # Simulate smooth progress based on time
        start_time = time.time()
        expected_duration = 240  # 4 minutes expected
        timeout = 900  # 15 minute timeout (increased from 10)
        
        while True:
            # Check if process finished
            retcode = process.poll()
            if retcode is not None:
                # Process finished
                stdout, stderr = process.communicate()
                
                if retcode == 0:
                    # Success
                    print("Backtest completed successfully!")
                    
                    results_dir = Path('backtest_results')
                    summary_file = results_dir / 'summary.json'
                    
                    if summary_file.exists():
                        with open(summary_file, 'r') as f:
                            results = json.load(f)
                        
                        backtest_status['results'] = {
                            'total_trades': results.get('total_trades', 0),
                            'winrate': results.get('winrate', 0) * 100,
                            'total_pnl': results.get('final_equity', 10000) - initial_capital,
                            'sharpe_ratio': results.get('sharpe_ratio', 0),
                            'max_drawdown': results.get('max_drawdown', 0) * 100,
                            'profit_factor': results.get('profit_factor', 0)
                        }
                    else:
                        backtest_status['results'] = {'error': 'Results file not found'}
                else:
                    # Failed
                    error_msg = stderr or stdout
                    print(f"Backtest failed: {error_msg[:500]}")
                    backtest_status['results'] = {'error': f'Failed: {error_msg[:200]}'}
                
                backtest_status['progress'] = 100
                break
            
            # Update progress based on elapsed time
            elapsed = time.time() - start_time
            
            # Check timeout
            if elapsed > timeout:
                process.kill()
                backtest_status['results'] = {'error': 'Timeout (>10 min)'}
                backtest_status['progress'] = 100
                break
            
            # Smooth progress: 0-80% over expected duration
            progress = min(int((elapsed / expected_duration) * 80), 80)
            backtest_status['progress'] = progress
            
            time.sleep(1)  # Update every second
        
    except Exception as e:
        print(f"Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        backtest_status['results'] = {'error': str(e)}
        backtest_status['progress'] = 100
    
    finally:
        backtest_status['running'] = False
        print(f"Backtest finished. Final status: {backtest_status}")

@app.route('/api/trades/execute', methods=['POST'])
def execute_trade():
    """
    Execute a trade
    
    Request body:
    {
        "symbol": "EURUSD",
        "side": "BUY",
        "size": 1000,
        "sl": 1.0650,
        "tp": 1.0720
    }
    """
    data = request.json
    
    # In a real implementation, this would execute the trade
    # For now, just return success
    return jsonify({
        'status': 'executed',
        'trade_id': 'TRADE_' + datetime.now().strftime('%Y%m%d%H%M%S'),
        'symbol': data.get('symbol'),
        'side': data.get('side'),
        'size': data.get('size'),
        'sl': data.get('sl'),
        'tp': data.get('tp'),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 SCOPUS Jarvis HUD - Backend API Server")
    print("=" * 60)
    print(f"Starting server at http://localhost:5000")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print("\nAvailable endpoints:")
    print("  GET  /api/health              - Health check")
    print("  GET  /api/agent/status        - Agent status")
    print("  GET  /api/market/overview     - Market data")
    print("  POST /api/backtest/run        - Run backtest")
    print("  GET  /api/backtest/status     - Backtest progress")
    print("  GET  /api/backtest/results    - Backtest results")
    print("  POST /api/trades/execute      - Execute trade")
    print("=" * 60)
    print("\nPress Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
