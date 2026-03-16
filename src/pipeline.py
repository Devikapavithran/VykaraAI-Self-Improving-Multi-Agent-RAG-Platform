
import time
import uuid
import re
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from agents.query_agent import rewrite_query
from retrieval.hybrid_search import hybrid_search
from retrieval.reranker import rerank
from agents.evaluator import evaluate_answer

from src.utils.logger import logger


# --------------------------------------------------
# LOAD EMBEDDINGS + FAISS (Loaded Once)
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
FAISS_PATH = os.path.abspath("faiss_index")
db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# --------------------------------------------------
# COMPANY DETECTION
# --------------------------------------------------

def detect_companies(query: str):
    q = query.lower()
    companies = []

    if "amazon" in q:
        companies.append("AMAZON")
    if "nvidia" in q:
        companies.append("NVIDIA")
    if "apple" in q:
        companies.append("APPLE")
    if "jpmorgan" in q:
        companies.append("JPMORGAN")

    return companies


# --------------------------------------------------
# QUERY BOOSTING
# --------------------------------------------------

def boost_query(query: str):
    q = query.lower()

    if "risk" in q:
        return query + " Item 1A Risk Factors"
    if "revenue" in q:
        return query + " Consolidated Statements of Operations Net Sales"
    if "income" in q:
        return query + " Net Income Consolidated Financial Statements"

    return query


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------

def run_vykara(query: str):

    start_time = time.time()
    request_id = str(uuid.uuid4())

    logger.info(f"[{request_id}] Incoming query: {query}")

    try:

        # ------------------------------------------
        # QUERY REWRITE
        # ------------------------------------------

        optimized_query = rewrite_query(query)
        logger.info(f"[{request_id}] Optimized query: {optimized_query}")

        boosted_query = boost_query(optimized_query)
        companies = detect_companies(optimized_query)

        is_comparison = len(companies) >= 2

        # ------------------------------------------
        # RETRIEVAL
        # ------------------------------------------

        docs = hybrid_search(boosted_query, db, k=8)

        if not docs:
            logger.warning(f"[{request_id}] No documents retrieved.")

            return {
                "request_id": request_id,
                "query": query,
                "analysis": {},
                "comparison_summary": [],
                "sources": [],
                "confidence_score": 0.1,
                "evaluation": {
                    "score": 1,
                    "verdict": "BAD",
                    "reason": "No relevant documents retrieved."
                },
                "retrieval": {
                    "chunks_retrieved": 0,
                    "chunks_after_rerank": 0,
                    "retry_triggered": False
                },
                "latency_seconds": round(time.time() - start_time, 2)
            }

        logger.info(f"[{request_id}] Retrieved {len(docs)} chunks")

        docs = rerank(optimized_query, docs, top_k=5)

        logger.info(f"[{request_id}] After rerank: {len(docs)} chunks")

        context = "\n\n".join([d.page_content for d in docs])

        sources = list(set(
            d.metadata.get("source")
            for d in docs if "source" in d.metadata
        ))

        # ------------------------------------------
        # PROMPT CREATION
        # ------------------------------------------

        if is_comparison:
            prompt = f"""
You are a senior financial analyst.

Use ONLY the provided context.

Answer strictly in this structure:

COMPANY NAME:
- Bullet
- Bullet

COMPANY NAME:
- Bullet
- Bullet

COMPARISON:
- Bullet
- Bullet

Rules:
- Bullets must start with "-"
- No paragraphs
- No tables

Context:
{context}

Question:
{query}
"""
        else:
            prompt = f"""
You are a senior financial analyst.

Use ONLY the provided context.

Answer strictly in this structure:

- Bullet
- Bullet

Rules:
- Bullets must start with "-"
- No paragraphs
- No tables

Context:
{context}

Question:
{query}
"""

        # ------------------------------------------
        # LLM CALL
        # ------------------------------------------

        response = llm.invoke(prompt).content.strip()

        # ------------------------------------------
        # STRUCTURED PARSING
        # ------------------------------------------

        structured_analysis = {}
        comparison_summary = []

        if is_comparison:
            current_company = None

            for line in response.split("\n"):
                clean = line.strip()

                if not clean:
                    continue

                header_match = re.match(r"^[A-Z\s]+:$", clean)
                if header_match:
                    current_company = clean.replace(":", "").strip()
                    structured_analysis[current_company] = []
                    continue

                if clean.startswith("-") and current_company:
                    structured_analysis[current_company].append(clean)

            if "COMPARISON" in structured_analysis:
                comparison_summary = structured_analysis.pop("COMPARISON")

        else:
            lines = [l.strip() for l in response.split("\n") if l.strip()]
            bullets = [l for l in lines if l.startswith("-")]

            if not bullets:
                bullets = [f"- {response}"]

            key = companies[0] if companies else "GENERAL"
            structured_analysis[key] = bullets

        # ------------------------------------------
        # EVALUATION
        # ------------------------------------------

        evaluation = evaluate_answer(query, context, response)
        confidence = round(evaluation["score"] / 10, 2)

        logger.info(
            f"[{request_id}] Evaluation score: {evaluation['score']} | "
            f"Confidence: {confidence}"
        )

        retry_triggered = False

        if evaluation["score"] <= 4:
            retry_triggered = True
            logger.info(f"[{request_id}] Retry triggered")

            docs_retry = hybrid_search(boosted_query, db, k=12)
            docs_retry = rerank(optimized_query, docs_retry, top_k=8)

            context_retry = "\n\n".join([d.page_content for d in docs_retry])
            response_retry = llm.invoke(prompt).content.strip()

            evaluation_retry = evaluate_answer(query, context_retry, response_retry)

            if evaluation_retry["score"] > evaluation["score"]:
                response = response_retry
                evaluation = evaluation_retry
                confidence = round(evaluation["score"] / 10, 2)

        latency = round(time.time() - start_time, 2)

        logger.info(f"[{request_id}] Completed in {latency}s")

        # ------------------------------------------
        # FINAL RETURN
        # ------------------------------------------

        return {
            "request_id": request_id,
            "query": query,
            "analysis": structured_analysis,
            "comparison_summary": comparison_summary,
            "sources": sources,
            "confidence_score": confidence,
            "evaluation": evaluation,
            "retrieval": {
                "chunks_retrieved": 8,
                "chunks_after_rerank": 5,
                "retry_triggered": retry_triggered
            },
            "latency_seconds": latency
        }

    except Exception as e:
        logger.error(f"[{request_id}] ERROR: {str(e)}")
        raise