import time
from packages.agents.reliability import retry_with_backoff

class _Flaky:
    def __init__(self, fail_times=2):
        self.calls = 0
        self.fail_times = fail_times

    def run(self):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise Exception("transient")
        return "ok"

def test_retry_succeeds_after_retries():
    f = _Flaky(fail_times=2)
    @retry_with_backoff(retries=3, backoff_in_seconds=0)  # fast test
    def wrapped():
        return f.run()
    assert wrapped() == "ok"
