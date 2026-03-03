from sentence_transformers import CrossEncoder

# elite reranker model (used in real systems)
reranker_model = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def rerank(query, docs, top_k=4):

    pairs = [(query, d.page_content) for d in docs]

    scores = reranker_model.predict(pairs)

    ranked_docs = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, _ in ranked_docs[:top_k]]
