from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


# LLM for rewriting queries
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",   # stable + cheap
    temperature=0.2
)


def rewrite_query(user_query: str) -> str:
    """
    Converts vague user questions into retrieval-optimized queries.
    """

    system_prompt = """
You are a query rewriting expert for a financial document retrieval system.

Rules:
- Make the query specific
- Expand abbreviations
- Add financial context if missing
- Keep meaning unchanged
- Output ONLY the rewritten query
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ]

    response = llm.invoke(messages)

    return response.content.strip()
