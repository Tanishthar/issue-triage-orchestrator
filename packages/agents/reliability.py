import time
import json
import os
from functools import wraps
from datetime import datetime

# === Retry with exponential backoff ===
def retry_with_backoff(retries=3, backoff_in_seconds=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt < retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempt += 1
                    wait_time = backoff_in_seconds * (2 ** (attempt - 1))
                    log_step(func.__name__, f"Attempt {attempt} failed: {e}, retrying in {wait_time}s")
                    time.sleep(wait_time)
            raise Exception(f"Function {func.__name__} failed after {retries} retries")
        return wrapper
    return decorator


# === Circuit Breaker ===
class CircuitBreaker:
    def __init__(self, failure_threshold=3, recovery_time=30):
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time
        self.failure_count = 0
        self.last_failure_time = None
        self.open = False

    def before_call(self, func_name):
        if self.open and (time.time() - self.last_failure_time < self.recovery_time):
            raise Exception(f"Circuit open: Skipping {func_name}")
        elif self.open:
            self.open = False
            self.failure_count = 0

    def record_failure(self, func_name):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.open = True
            log_step(func_name, "⚠️ Circuit breaker OPEN")

    def record_success(self, func_name):
        self.failure_count = 0
        self.open = False


# === Step Logger ===
LOG_FILE = "metrics/step_logs.json"

def log_step(step_name, message):
    os.makedirs("metrics", exist_ok=True)
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "step": step_name,
        "message": message,
    }
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    else:
        logs = []
    logs.append(log_entry)
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)
