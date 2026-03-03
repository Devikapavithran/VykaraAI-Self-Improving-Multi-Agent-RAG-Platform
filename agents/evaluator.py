import json
from langchain_google_genai import ChatGoogleGenerativeAI


# Initialize evaluator LLM separately (stable temperature)
evaluator_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0
)


def evaluate_answer(question, context, answer):
    """
    Evaluates answer quality and grounding.
    Always returns valid structured JSON.
    Never crashes.
    """

    evaluation_prompt = f"""
You are an evaluation agent.

Evaluate the quality of the answer strictly based on the provided context.

Scoring Criteria:
- 9-10: Fully grounded, accurate, complete
- 7-8: Mostly grounded, minor issues
- 5-6: Partially grounded or incomplete
- 1-4: Hallucinated, incorrect, or not grounded

Return ONLY valid JSON in this exact format:

{{
  "score": <integer between 1 and 10>,
  "verdict": "GOOD" or "BAD",
  "reason": "<short explanation>"
}}

Context:
{context}

Question:
{question}

Answer:
{answer}
"""

    try:
        raw_response = evaluator_llm.invoke(evaluation_prompt).content.strip()

        # Attempt to extract JSON safely
        start = raw_response.find("{")
        end = raw_response.rfind("}") + 1
        json_string = raw_response[start:end]

        parsed = json.loads(json_string)

        # Safety normalization
        score = int(parsed.get("score", 6))
        verdict = parsed.get("verdict", "GOOD")
        reason = parsed.get("reason", "Evaluation fallback")

        return {
            "score": score,
            "verdict": verdict,
            "reason": reason
        }

    except Exception:
        # Hard fallback (never fail)
        return {
            "score": 6,
            "verdict": "GOOD",
            "reason": "Evaluation parsing fallback"
        }
