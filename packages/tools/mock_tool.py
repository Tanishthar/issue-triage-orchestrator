import random
from packages.agents.reliability import retry_with_backoff, CircuitBreaker, log_step

breaker = CircuitBreaker()

@retry_with_backoff(retries=3, backoff_in_seconds=1)
def mock_classify_issue(issue_text: str):
    breaker.before_call("mock_classify_issue")

    if random.random() < 0.3:
        breaker.record_failure("mock_classify_issue")
        raise Exception("Random tool error")

    severity = random.choice(["low", "medium", "high"])
    breaker.record_success("mock_classify_issue")

    log_step("mock_classify_issue", f"Issue classified as {severity}")
    return severity
