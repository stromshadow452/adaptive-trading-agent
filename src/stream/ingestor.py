import zmq
import sqlite3
import json
import time
import threading
from datetime import datetime, timezone

class StreamIngestor:
    """
    Stage 1: Market Data -> Live Stream Reactor
    Ingests ticks, persists to WAL (SQLite), and pushes to ZeroMQ.
    Implements:
    - Seq-ID for gap detection
    - Heartbeat every 1s
    - WAL Persistence
    - Replay capability
    """
    def __init__(self, db_path="data/stream.db", zmq_port=5555):
        self.ctx = zmq.Context()
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.bind(f"tcp://*:{zmq_port}")
        
        self.db_path = db_path
        self._init_db()
        
        self.seq_id = self._get_last_seq()
        self.running = True
        
        # Start Heartbeat Thread
        self.hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.hb_thread.start()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ticks (
                seq INTEGER PRIMARY KEY,
                ts REAL, 
                symbol TEXT, 
                price REAL, 
                vol REAL,
                iso TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _get_last_seq(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.execute("SELECT MAX(seq) FROM ticks")
        res = cur.fetchone()[0]
        conn.close()
        return res if res else 0

    def ingest(self, symbol: str, price: float, volume: float):
        ts = time.time()
        iso = datetime.now(timezone.utc).isoformat()
        self.seq_id += 1
        
        # 1. Persist (WAL)
        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT INTO ticks VALUES (?, ?, ?, ?, ?, ?)", 
                     (self.seq_id, ts, symbol, price, volume, iso))
        conn.commit()
        conn.close()
        
        # 2. Publish (ZeroMQ)
        payload = {
            "seq": self.seq_id,
            "ts": ts,
            "symbol": symbol,
            "price": price,
            "vol": volume,
            "iso": iso
        }
        self.pub.send_string(f"{symbol} {json.dumps(payload)}")

    def _heartbeat_loop(self):
        while self.running:
            time.sleep(1)
            payload = {
                "seq": self.seq_id, # Current max seq
                "ts": time.time(),
                "type": "HEARTBEAT"
            }
            self.pub.send_string(f"__HEARTBEAT__ {json.dumps(payload)}")

    def replay_range(self, from_seq: int, to_seq: int):
        """Replay missing sequence range"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.execute(
            "SELECT symbol, price, vol, ts, iso, seq FROM ticks WHERE seq >= ? AND seq <= ? ORDER BY seq ASC", 
            (from_seq, to_seq)
        )
        rows = cur.fetchall()
        conn.close()
        
        replayed = []
        for r in rows:
            replayed.append({
                "symbol": r[0],
                "price": r[1],
                "vol": r[2],
                "ts": r[3],
                "iso": r[4],
                "seq": r[5]
            })
        return replayed

    def stop(self):
        self.running = False
        self.ctx.term()
