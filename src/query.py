from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from src.pipeline import run_vykara
import time
import logging
import os

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
# BASIC METRICS
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
    x_api_key: str = Header(None)
):

    expected_key = os.getenv("DEMO_API_KEY")

    if x_api_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Unauthorized. Invalid API key."
        )

    global total_requests
    global total_latency

    start_time = time.time()

    result = run_vykara(request.query)

    latency = time.time() - start_time

    total_requests += 1
    total_latency += latency

    logging.info(
        f"Query: {request.query} | "
        f"Confidence: {result.get('confidence_score')} | "
        f"Latency: {result.get('latency_seconds')}s | "
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
        "average_latency_seconds": average_latency
    }