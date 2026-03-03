from langchain_community.retrievers import BM25Retriever


def hybrid_search(query, db, k=8, company=None):
    """
    Hybrid retrieval with source-based filtering.
    Compatible with newer LangChain versions.
    """

    # 🔥 Step 1: Broad dense retrieval
    all_docs = db.similarity_search(query, k=50)

    # 🔥 Step 2: Filter by company using source filename
    if company:
        company = company.lower()
        filtered_docs = []

        for d in all_docs:
            source = d.metadata.get("source", "").lower()
            if company in source:
                filtered_docs.append(d)

        all_docs = filtered_docs

    if not all_docs:
        return []

    # 🔥 Step 3: Dense top-k from filtered pool
    dense_docs = all_docs[:k]

    # 🔥 Step 4: BM25 keyword retrieval
    bm25 = BM25Retriever.from_documents(all_docs)
    bm25.k = k

    # NEW API STYLE
    keyword_docs = bm25.invoke(query)

    # 🔥 Step 5: Merge and deduplicate
    combined = dense_docs + keyword_docs

    seen = set()
    unique_docs = []

    for doc in combined:
        if doc.page_content not in seen:
            unique_docs.append(doc)
            seen.add(doc.page_content)

    return unique_docs
