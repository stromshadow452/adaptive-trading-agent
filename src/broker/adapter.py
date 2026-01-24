import time

class BrokerAdapter:
    """
    Stage 11: Broker + Hedger -> Multi-Broker Bridge
    """
    def __init__(self, broker_name="paper"):
        self.broker_name = broker_name
        self.latency_guard_ms = 500

    def get_price(self, symbol):
        t0 = time.perf_counter()
        # Mock API Call
        price = 1.1000 
        latency = (time.perf_counter() - t0) * 1000
        
        if latency > self.latency_guard_ms:
            print(f"[WARN] Broker Latency High: {latency:.2f}ms")
            # In prod: pause trading
            
        return price

    def execute(self, symbol, side, size):
        print(f"[{self.broker_name}] EXECUTE {side} {size} {symbol}")
        return True
