from fastapi import FastAPI, HTTPException, Header, Request
from pydantic import BaseModel
from src.pipeline import run_vykara
import time
import logging
import os

# -----------------------
# CONFIG
# -----------------------

MAX_DEMO_REQUESTS = 50
RATE_LIMIT_SECONDS = 5
request_tracker = {}

# -----------------------
# LOGGING CONFIG
# -----------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

app = FastAPI(
    title="Vykara API",
    description="Production-Grade Agentic RAG Infrastructure",
    version="1.0"
)

# -----------------------
# REQUEST MODEL
# -----------------------

class QueryRequest(BaseModel):
    query: str

# -----------------------
# METRICS
# -----------------------

total_requests = 0
total_latency = 0

# -----------------------
# HEALTH CHECK
# -----------------------

@app.get("/health")
def health_check():
    return {"status": "Vykara is running"}

# -----------------------
# QUERY ENDPOINT (Protected)
# -----------------------

@app.post("/query")
def query_vykara(
    request: QueryRequest,
    http_request: Request,
    x_api_key: str = Header(None)
):

    global total_requests
    global total_latency

    # 🔐 Demo API key protection
    expected_key = os.getenv("DEMO_API_KEY")

    if x_api_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized. Invalid API key."
        )

    # 🛑 Hard demo cap protection
    if total_requests >= MAX_DEMO_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Demo request limit reached."
        )

    # ⏳ Basic per-IP cooldown
    client_ip = http_request.client.host
    current_time = time.time()

    if client_ip in request_tracker:
        last_time = request_tracker[client_ip]
        if current_time - last_time < RATE_LIMIT_SECONDS:
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please wait a few seconds."
            )

    request_tracker[client_ip] = current_time

    # 🚀 Run pipeline
    start_time = time.time()
    result = run_vykara(request.query)
    latency = time.time() - start_time

    total_requests += 1
    total_latency += latency

    logging.info(
        f"Query: {request.query} | "
        f"Confidence: {result.get('confidence_score')} | "
        f"Latency: {latency:.2f}s | "
        f"Sources Used: {len(result.get('sources', []))}"
    )

    return result

# -----------------------
# METRICS ENDPOINT
# -----------------------

@app.get("/metrics")
def metrics():

    average_latency = (
        round(total_latency / total_requests, 2)
        if total_requests > 0 else 0
    )

    return {
        "total_requests": total_requests,
        "average_latency_seconds": average_latency,
        "max_demo_requests": MAX_DEMO_REQUESTS
    }