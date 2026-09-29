"""
Real API test for ProductionAgent (Gemini version).
Run from the project root:  uv run python test_agent.py
"""

import json

from langchain_google_genai import ChatGoogleGenerativeAI

from app.agent import ProductionAgent          # change if your file has a different name
from app.config import get_settings
from app.monitoring import get_logger, MetricsCollector, RequestTimer

settings = get_settings()
logger = get_logger("agent-test")
metrics = MetricsCollector()

QUESTION = "Reply with exactly one word: what is the capital of Japan?"
results = []


def broken_llm():
    """A real Gemini client pointed at a model that does not exist."""
    return ChatGoogleGenerativeAI(
        model="this-model-does-not-exist",
        temperature=0,
        timeout=30,
        max_retries=0,
        api_key=settings.gemini_api_key,
    )


def slow_llm():
    """A real client for the primary model with an impossibly short timeout."""
    return ChatGoogleGenerativeAI(
        model=settings.primary_model,
        temperature=0,
        timeout=0.001,
        max_retries=0,
        api_key=settings.gemini_api_key,
    )


def run(name, expected_model, setup=None):
    agent = ProductionAgent()
    if setup:
        setup(agent)

    print(f"\n=== {name} ===")
    print(f"primary={agent.primary_llm.model}  "
          f"fallback={agent.fallback_llm.model}  "
          f"max_retries={agent.max_retries}")

    with RequestTimer() as timer:
        out = agent.invoke(QUESTION)

    passed = out["model_used"] == expected_model
    metrics.record_request(
        latency_ms=timer.elapsed_ms,
        error=out["model_used"] == "error_handler",
    )
    logger.info(name, extra={"extra_data": {
        "expected": expected_model,
        "got": out["model_used"],
        "latency_ms": round(timer.elapsed_ms, 1),
        "passed": passed,
    }})

    print(f"response   : {out['response']}")
    print(f"model_used : {out['model_used']}  (expected {expected_model})")
    print(f"error      : {(out['error'] or '')[:150]}")
    print(f"time       : {timer.elapsed_ms:.0f} ms")
    print("PASS" if passed else "FAIL")
    results.append((name, passed))


# 1. Everything works
run("1. Normal call", "primary")


# 2. Primary broken, fallback allowed
def primary_broken_fallback_allowed(agent):
    agent.primary_llm = broken_llm()
    agent.max_retries = 2

run("2. Primary broken, max_retries=2", "fallback", primary_broken_fallback_allowed)


# 3. Primary broken, max_retries=1 (routing skips the fallback)
def primary_broken_no_fallback(agent):
    agent.primary_llm = broken_llm()
    agent.max_retries = 1

run("3. Primary broken, max_retries=1", "error_handler", primary_broken_no_fallback)


# 4. Both broken
def both_broken(agent):
    agent.primary_llm = broken_llm()
    agent.fallback_llm = broken_llm()
    agent.max_retries = 2

run("4. Both models broken", "error_handler", both_broken)


# 5. Primary times out
def primary_timeout(agent):
    agent.primary_llm = slow_llm()
    agent.max_retries = 2

run("5. Primary times out", "fallback", primary_timeout)


# Summary
print("\n=== RESULTS ===")
for name, passed in results:
    print(f"{'PASS' if passed else 'FAIL'}  {name}")
print(f"\n{sum(p for _, p in results)}/{len(results)} passed")

print("\n=== METRICS ===")
print(json.dumps(metrics.summary, indent=2))