"""
Production-Ready FastAPI + LangGraph Application

Wires together:
- Security pipeline (input sanitization, PII masking)
- Response caching
- Rate limiting (slowapi)
- LangGraph agent (with retries + fallback)
- Structured logging + metrics
- LangSmith tracing
- Health checks
"""

import time
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from langsmith import traceable
from dotenv import load_dotenv

from app.config import get_settings
from app.models import (
    ChatRequest, ChatResponse,
    HealthResponse, MetricsResponse, ErrorResponse,
)
from app.security import SecurityPipeline
from app.cache import ResponseCache
from app.monitoring import get_logger, MetricsCollector, RequestTimer
from app.agent import ProductionAgent

load_dotenv()



# === Global instances (initialized in lifespan) ===
security: SecurityPipeline = None
cache: ResponseCache = None
metrics: MetricsCollector = None
agent: ProductionAgent = None
logger = get_logger()


# === Lifespan (startup/shutdown) ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize all components on startup, clean up on shutdown.
    This is the modern FastAPI pattern (replaces @app.on_event).
    """
    global security, cache, metrics, agent

    settings = get_settings()

    logger.info("Starting production API...", extra={"extra_data": {
        "environment": settings.app_env,
        "primary_model": settings.primary_model,
        "tracing_enabled": settings.langchain_tracing,
    }})

    # Initialize components
    security = SecurityPipeline()
    cache = ResponseCache(ttl_seconds=settings.cache_ttl_seconds)
    metrics = MetricsCollector()
    agent = ProductionAgent()

    logger.info("All components initialized. Ready to serve requests.")

    yield  # App is running

    # Shutdown
    logger.info("Shutting down...", extra={"extra_data": metrics.summary})
    # On shutdown, log the final scoreboard. Since metrics live in memory, this is the last chance to see them.
    
    # === Rate Limiter Setup ===
limiter = Limiter(key_func=get_remote_address)

# === FastAPI App ===
app = FastAPI(
    title="Production LangGraph API",
    description="A production-ready chat API with security, caching, and observability.",
    version="1.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
# app.state is a storage shelf on the app. 
# slowapi looks for the limiter there, so this line is required for rate limiting to work.


# === Exception Handlers ===

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """
    Handle rate limit exceeded errors.
    "Whenever a RateLimitExceeded error happens anywhere, run this function instead of crashing.
    """
    logger.warning("Rate limit exceeded", extra={"extra_data": {
        "client_ip": get_remote_address(request),
    }})
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": "Too many requests. Please slow down.",
        },
    )
    

# =============================================
# ENDPOINTS
# =============================================

@app.post("/chat", response_model=ChatResponse)
@limiter.limit(get_settings().rate_limit)
@traceable(name="chat_endpoint")
async def chat(request: Request, body: ChatRequest):
    """
    Main chat endpoint.

    Flow:
    1. Security check (injection + PII masking)
    2. Cache lookup
    3. LangGraph agent invoke (if cache miss)
    4. Output validation
    5. Cache store
    6. Return response
    """
    with RequestTimer() as timer:
        security_notes = []

        # ---- Step 1: Security Check ----
        is_allowed, cleaned_message, notes = security.check_input(body.message)
        security_notes.extend(notes)

        if not is_allowed:
            logger.warning("Request blocked by security", extra={"extra_data": {
                "reason": notes,
                "thread_id": body.thread_id,
            }})
            metrics.record_request(latency_ms=0, error=True)
            raise HTTPException(
                status_code=400,
                detail="Your message was blocked by our security filters."
            )

        # ---- Step 2: Cache Lookup ----
        cached_response = cache.get(cleaned_message)
        if cached_response is not None:
            metrics.record_request(latency_ms=0, cache_hit=True)
            logger.info("Cache hit", extra={"extra_data": {
                "thread_id": body.thread_id,
            }})
            return ChatResponse(
                response=cached_response,
                thread_id=body.thread_id,
                model_used="cache",
                cached=True,
                processing_time_ms=0,
            )

        # ---- Step 3: Invoke LangGraph Agent ----
        try:
            result = agent.invoke(cleaned_message)
        except Exception as e:
            logger.error(f"Agent invocation failed: {e}", extra={"extra_data": {
                "thread_id": body.thread_id,
                "error": str(e),
            }})
            metrics.record_request(latency_ms=0, error=True)
            raise HTTPException(
                status_code=500,
                detail="An error occurred while processing your request."
            )

        response_text = result["response"]
        model_used = result["model_used"]

        # ---- Step 4: Output Validation ----
        validated_response, output_warnings = security.check_output(response_text)
        security_notes.extend(output_warnings)

        # ---- Step 5: Cache Store ----
        cache.set(cleaned_message, validated_response)

    # ---- Step 6: Log & Record Metrics ----
    input_tokens = int(len(cleaned_message.split()) * 1.3)
    output_tokens = int(len(validated_response.split()) * 1.3)

    metrics.record_request(
        latency_ms=timer.elapsed_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_hit=False,
    )

    if security_notes:
        logger.info("Security notes", extra={"extra_data": {
            "notes": security_notes,
            "thread_id": body.thread_id,
        }})

    logger.info("Request completed", extra={"extra_data": {
        "thread_id": body.thread_id,
        "model_used": model_used,
        "latency_ms": round(timer.elapsed_ms, 2),
    }})

    return ChatResponse(
        response=validated_response,
        thread_id=body.thread_id,
        model_used=model_used,
        cached=False,
        processing_time_ms=round(timer.elapsed_ms, 2),
        security_notes=security_notes,
    )
    
    
    
    
@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check for Docker/Kubernetes."""
    settings = get_settings()

    checks = {
        "agent": agent is not None,
        "security": security is not None,
        "cache": cache is not None,
    }

    all_healthy = all(checks.values())

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        environment=settings.app_env,
        checks=checks,
    )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Metrics for monitoring dashboards."""
    summary = metrics.summary
    return MetricsResponse(**summary)


@app.get("/cache/stats")
async def cache_stats():
    """Cache performance statistics."""
    return cache.stats





"""
code explained:
## The big picture first

The last two files were **parts**: a diary and scoreboard (monitoring), and a brain with a backup (the agent). This file is the **building that puts them together** and opens the doors to the public.

Think of it as the front desk of an office:

1. **Security guard** checks every visitor (blocks attacks, hides personal info).
2. **Receptionist with a notebook** checks: "Has someone asked this exact question before? Here's the saved answer." (cache)
3. **Door limit**: one person can't walk in 100 times a minute. (rate limiting)
4. If it's a new question, it goes to the **expert** (the LangGraph agent).
5. The answer is **checked again** before handing it back (output validation).
6. Everything is **written in the diary and scoreboard** (logging and metrics).

It uses **FastAPI**, a Python framework that turns functions into web addresses (endpoints) that apps and websites can call.

I haven't seen `app/security.py`, `app/cache.py` or `app/models.py`, so for those I'll explain what they do based on how this file uses them.

---

## Part 0: The top docstring

The text in triple quotes at the top is just a description. Python ignores it when running. It lists what this file wires together.

---

## Part 1: Imports

```python
import time
import os
```
Standard Python tools. Neither is actually used in this file, so they can be removed.

```python
from contextlib import asynccontextmanager
```
A helper for writing "setup, then run, then clean up" functions. Used for `lifespan` below.

```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
```
- `FastAPI`: the web application itself.
- `Request`: details about an incoming call (who sent it, headers, IP address).
- `HTTPException`: a way to reply with an error code, like 400 or 500.
- `JSONResponse`: a way to build a custom JSON reply.

```python
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
```
**slowapi** is a rate-limiting library.
- `Limiter`: the bouncer that counts requests.
- `get_remote_address`: gets the caller's IP address, used to tell callers apart.
- `RateLimitExceeded`: the error raised when someone goes over the limit.

```python
from langsmith import traceable
from dotenv import load_dotenv
```
Tracing (from earlier), and the tool that loads `.env` into the environment.

```python
from app.config import get_settings
from app.models import (
    ChatRequest, ChatResponse,
    HealthResponse, MetricsResponse, ErrorResponse,
)
```
Your settings, and **models** (Pydantic classes) that describe the exact shape of requests and replies. For example, `ChatRequest` probably says "a request must have a `message` (text) and a `thread_id`". FastAPI uses these to automatically reject badly formed requests. `ErrorResponse` is imported but not used.

```python
from app.security import SecurityPipeline
from app.cache import ResponseCache
from app.monitoring import get_logger, MetricsCollector, RequestTimer
from app.agent import ProductionAgent
```
Your own building blocks: security, cache, the monitoring file you already know, and the agent.

```python
load_dotenv()
```
Copies `.env` values into the environment. This is what makes LangSmith tracing work here (the issue we discussed for the test script).

---

## Part 2: Global placeholders

```python
security: SecurityPipeline = None
cache: ResponseCache = None
metrics: MetricsCollector = None
agent: ProductionAgent = None
logger = get_logger()
```

Empty slots for the main components. They start as `None` and get filled in when the app starts (next part). The `: SecurityPipeline` part is only a label for readers and editors; it doesn't enforce anything.

The logger is created immediately because you want to log even during startup.

Why not create everything right here? Because creating the agent involves loading settings and connecting to Gemini. It's cleaner to do that at a controlled moment, when the server actually starts.

---

## Part 3: Lifespan (startup and shutdown)

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
```

This function runs **once** when the server starts and once when it stops. Analogy: opening and closing up a shop.

```python
    global security, cache, metrics, agent
```
"When I assign these names below, I mean the global slots from Part 2, not new local variables." Without this line, the assignments would disappear when the function ends.

```python
    settings = get_settings()

    logger.info("Starting production API...", extra={"extra_data": {
        "environment": settings.app_env,
        "primary_model": settings.primary_model,
        "tracing_enabled": settings.langchain_tracing,
    }})
```
Log a startup message, with which environment (dev/prod), which model, and whether tracing is on. That `extra_data` pattern is exactly what you learned in the monitoring file.

```python
    security = SecurityPipeline()
    cache = ResponseCache(ttl_seconds=settings.cache_ttl_seconds)
    metrics = MetricsCollector()
    agent = ProductionAgent()
```
Build each component once. `ttl_seconds` means **time to live**: how long a cached answer stays valid before it's thrown away.

```python
    logger.info("All components initialized. Ready to serve requests.")

    yield  # App is running
```
`yield` is the dividing line. Everything **before** it runs at startup. Then the server runs and handles requests for as long as it's up. Everything **after** it runs at shutdown.

```python
    logger.info("Shutting down...", extra={"extra_data": metrics.summary})
```
On shutdown, log the final scoreboard. Since metrics live in memory, this is the last chance to see them.

---

## Part 4: Rate limiter and the app

```python
limiter = Limiter(key_func=get_remote_address)
```
Create the bouncer. `key_func` decides **how to tell callers apart**. Here it's by IP address, so each IP gets its own counter.

```python
app = FastAPI(
    title="Production LangGraph API",
    description="A production-ready chat API with security, caching, and observability.",
    version="1.0.0",
    lifespan=lifespan,
)
```
Create the web app. The title, description and version appear on FastAPI's automatic documentation page (visit `/docs` in a browser). `lifespan=lifespan` connects the startup/shutdown function.

```python
app.state.limiter = limiter
```
`app.state` is a storage shelf on the app. slowapi looks for the limiter there, so this line is required for rate limiting to work.

---

## Part 5: Rate limit error handler

```python
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
```
"Whenever a `RateLimitExceeded` error happens anywhere, run this function instead of crashing."

```python
    logger.warning("Rate limit exceeded", extra={"extra_data": {
        "client_ip": get_remote_address(request),
    }})
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": "Too many requests. Please slow down.",
        },
    )
```
Log who got blocked, and reply with **429**, the standard code meaning "too many requests".

---

## Part 6: The `/chat` endpoint (the main part)

```python
@app.post("/chat", response_model=ChatResponse)
@limiter.limit(get_settings().rate_limit)
@traceable(name="chat_endpoint")
async def chat(request: Request, body: ChatRequest):
```

Three decorators stacked, read from the bottom up:

- **`@traceable`**: record this function in LangSmith. Because `agent.invoke` is also traceable, the agent's trace appears **nested inside** this one. So in LangSmith you see the whole request, from front door to answer.
- **`@limiter.limit(...)`**: apply the rate limit, e.g. `"20/minute"` from your settings. slowapi needs the `request: Request` parameter to find the IP, which is why it's there even though the function doesn't otherwise use it.
- **`@app.post("/chat", ...)`**: register this function as the handler for POST requests to `/chat`. `response_model=ChatResponse` makes FastAPI check and shape the reply.

`body: ChatRequest`: FastAPI reads the JSON the caller sent, checks it matches `ChatRequest`, and hands it over as `body`. If it doesn't match, the caller gets a 422 error automatically.

`async`: this function can run alongside others without waiting for each other. (Important caveat later.)

```python
    with RequestTimer() as timer:
        security_notes = []
```
Start the stopwatch for the whole request, and make an empty list to collect any security remarks.

### Step 1: Security check

```python
        is_allowed, cleaned_message, notes = security.check_input(body.message)
        security_notes.extend(notes)
```
Hand the message to the security guard. It returns three things:
- `is_allowed`: should this message be processed at all? (False for things like prompt injection: "ignore your instructions and...")
- `cleaned_message`: the message with personal info masked, e.g. a phone number replaced with `[PHONE]`.
- `notes`: what the guard noticed.

`extend` adds those notes to the list.

```python
        if not is_allowed:
            logger.warning("Request blocked by security", extra={"extra_data": {
                "reason": notes,
                "thread_id": body.thread_id,
            }})
            metrics.record_request(latency_ms=0, error=True)
            raise HTTPException(
                status_code=400,
                detail="Your message was blocked by our security filters."
            )
```
If blocked: log why, count it as an error on the scoreboard, and reply with **400** (bad request). `raise` stops the function immediately, so nothing below runs.

### Step 2: Cache lookup

```python
        cached_response = cache.get(cleaned_message)
        if cached_response is not None:
```
"Has anyone asked this exact (cleaned) question recently?" Note it uses the **cleaned** message, so personal info never becomes part of the cache key.

```python
            metrics.record_request(latency_ms=0, cache_hit=True)
            logger.info("Cache hit", extra={"extra_data": {
                "thread_id": body.thread_id,
            }})
            return ChatResponse(
                response=cached_response,
                thread_id=body.thread_id,
                model_used="cache",
                cached=True,
                processing_time_ms=0,
            )
```
If found: record a cache hit, log it, and reply immediately with the saved answer. No AI call, so it's instant and free.

### Step 3: Call the agent

```python
        try:
            result = agent.invoke(cleaned_message)
        except Exception as e:
            logger.error(f"Agent invocation failed: {e}", extra={"extra_data": {
                "thread_id": body.thread_id,
                "error": str(e),
            }})
            metrics.record_request(latency_ms=0, error=True)
            raise HTTPException(
                status_code=500,
                detail="An error occurred while processing your request."
            )
```
Ask the agent. If something unexpected crashes it, log it, count it as an error, and reply with **500** (server error) and a safe message, without exposing technical details to the user.

```python
        response_text = result["response"]
        model_used = result["model_used"]
```
Pull out the answer and who produced it.

### Step 4: Output check

```python
        validated_response, output_warnings = security.check_output(response_text)
        security_notes.extend(output_warnings)
```
Check the AI's answer before sending it. For example, mask any personal info the model might have produced, or flag something inappropriate.

### Step 5: Save to cache

```python
        cache.set(cleaned_message, validated_response)
```
Store the answer so the next identical question is instant.

The `with` block ends here, so the stopwatch stops.

### Step 6: Record and reply

```python
    input_tokens = int(len(cleaned_message.split()) * 1.3)
    output_tokens = int(len(validated_response.split()) * 1.3)
```
A rough **estimate** of tokens: count words and multiply by 1.3 (on average, one English word is about 1.3 tokens). Not exact.

```python
    metrics.record_request(
        latency_ms=timer.elapsed_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_hit=False,
    )
```
Add this request to the scoreboard.

```python
    if security_notes:
        logger.info("Security notes", extra={"extra_data": {
            "notes": security_notes,
            "thread_id": body.thread_id,
        }})

    logger.info("Request completed", extra={"extra_data": {
        "thread_id": body.thread_id,
        "model_used": model_used,
        "latency_ms": round(timer.elapsed_ms, 2),
    }})
```
Log any security notes, then log that the request finished, with who answered and how long it took.

```python
    return ChatResponse(
        response=validated_response,
        thread_id=body.thread_id,
        model_used=model_used,
        cached=False,
        processing_time_ms=round(timer.elapsed_ms, 2),
        security_notes=security_notes,
    )
```
Send the final reply in the agreed shape.

---

## Part 7: Health, metrics and cache endpoints

```python
@app.get("/health", response_model=HealthResponse)
async def health():
    settings = get_settings()
    checks = {
        "agent": agent is not None,
        "security": security is not None,
        "cache": cache is not None,
    }
    all_healthy = all(checks.values())
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        environment=settings.app_env,
        checks=checks,
    )
```
Docker or Kubernetes calls `/health` regularly to ask "are you alive?". This checks each component was created. `all(...)` is True only if every check is True. If not, the status is "degraded", and the platform can restart the app.

```python
@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    summary = metrics.summary
    return MetricsResponse(**summary)
```
Return the scoreboard. `**summary` unpacks the dictionary into named arguments, e.g. `MetricsResponse(total_requests=5, total_errors=1, ...)`.

```python
@app.get("/cache/stats")
async def cache_stats():
    return cache.stats
```
Return the cache's own statistics.

---

## Things worth noticing

These go from most to least important.

**1. The apology gets cached.**
This connects to your test results. When Gemini returned 503, the agent returned the polite apology. This file treats that like a normal answer and **saves it in the cache**. For the next `cache_ttl_seconds`, anyone asking that question gets "I'm sorry, I'm having trouble...", even after Gemini recovers. Fix: skip caching when `model_used == "error_handler"`.

**2. Apologies count as successes.**
For the same reason, the scoreboard records an error-handler reply as a successful request. Your real error rate would look much better than it is. Fix: pass `error=(model_used == "error_handler")` to `record_request`.

**3. One slow request freezes the whole server.**
`chat` is `async`, but `agent.invoke` is a normal (blocking) function. While it waits for Gemini, the server can't do anything else. Your test showed a 26 second wait during the 503. During that time, **every** other user, and even `/health`, would be stuck. Kubernetes might then decide the app is dead and restart it. Fix:
```python
import asyncio
result = await asyncio.to_thread(agent.invoke, cleaned_message)
```
This runs the agent in a separate thread so the server stays responsive.

**4. The list response problem shows up here.**
From your test, Gemini returns `content` as a list. `security.check_output` and `.split()` expect text, so they'd likely crash or misbehave. The `.text` fix in the agent's `invoke` solves it here too.

**5. Latency is recorded as 0 on blocked, cached and failed requests.**
That drags the average down and makes the app look faster than it is. Using `timer.elapsed_ms` would be more honest, though for cache hits you might deliberately want to track them separately.

**6. Rate limiting by IP breaks behind a load balancer.**
On AWS, requests usually arrive through a load balancer, so every user appears to have the **load balancer's** IP. Everyone would share one limit. You'd need to read the real IP from the `X-Forwarded-For` header instead.

**7. `thread_id` is logged but not used for memory.**
It's passed around and logged, but never given to the agent. So the agent still has no conversation memory, as noted before. Also, the cache ignores `thread_id`, so the same question from different conversations gets the same answer. That's fine for standalone questions, but wrong once conversations have context ("what about the second one?").

**8. Everything is in memory.**
Cache, metrics and rate-limit counters all live inside one Python process. If you run several workers or containers, each has its own separate copy, and all of it resets on restart. Production setups typically move these to Redis (cache and rate limits) and Prometheus (metrics).

**9. The health check is shallow.**
It checks that objects exist, not that Gemini is reachable. That's actually a reasonable choice (you don't want Kubernetes restarting your app because Google is down), but worth knowing.

**10. Unused imports:** `time`, `os`, `ErrorResponse`.

---

## One-line summary

This file is the front door of your app: it checks each message for safety, returns saved answers when it can, limits how often each caller can ask, sends new questions to the agent, checks the answer, and records everything, with a few gaps (caching apologies, blocking calls) worth fixing before real traffic. 
"""