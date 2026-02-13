import time
from typing import Any, Dict, List

try:
    from .retriever import RAGRetriever
except ImportError:
    from retriever import RAGRetriever


def _extract_response_text(response: Any) -> str:
    if hasattr(response, "content"):
        return str(response.content)
    if hasattr(response, "output_text"):
        return str(response.output_text)
    if hasattr(response, "text"):
        return str(response.text)
    if isinstance(response, dict) and "output_text" in response:
        return str(response["output_text"])
    return str(response)


def rag_simple(query: str, retriever: RAGRetriever, llm: Any, top_k: int = 3) -> str:
    results = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join([doc["content"] for doc in results]) if results else ""
    if not context:
        return "No relevant context found to answer the question"

    prompt = f"""Use the following context to answer the question in detail.
Include definitions, key points, and practical examples where relevant.
Context:
{context}

Question:
{query}

Answer:
"""
    response = llm.invoke(prompt)
    return _extract_response_text(response)


def rag_advanced(
    query: str,
    retriever: RAGRetriever,
    llm: Any,
    top_k: int = 5,
    min_score: float = 0.2,
    return_context: bool = False,
) -> Dict[str, Any]:
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)
    if not results:
        return {
            "answer": "No relevant answer found",
            "sources": [],
            "confidence": 0.0,
            "context": "",
        }

    context = "\n\n".join([doc["content"] for doc in results])
    sources = [
        {
            "source": doc["metadata"].get(
                "source_file", doc["metadata"].get("source", "unknown")
            ),
            "page": doc["metadata"].get("page", "unknown"),
            "score": doc["similarity_score"],
            "preview": doc["content"][:120] + "...",
        }
        for doc in results
    ]
    confidence = max(doc["similarity_score"] for doc in results)

    prompt = f"""Use the following context to answer the question in detail.
Explain the topic clearly, include important subpoints, and provide concise examples.

Context:
{context}

Question: {query}

Answer:
"""
    response = llm.invoke(prompt)
    answer_text = _extract_response_text(response)
    output = {"answer": answer_text, "sources": sources, "confidence": confidence}
    if return_context:
        output["context"] = context
    return output


class AdvancedRagPipeline:
    def __init__(self, retriever: RAGRetriever, llm: Any):
        self.retriever = retriever
        self.llm = llm
        self.history: List[Dict[str, Any]] = []

    def _format_history(self, history_window: int = 4) -> str:
        if not self.history:
            return ""
        recent_turns = self.history[-history_window:]
        lines: List[str] = []
        for turn in recent_turns:
            q = turn.get("question", "").strip()
            a = turn.get("answer", "").strip()
            if q:
                lines.append(f"User: {q}")
            if a:
                lines.append(f"Assistant: {a}")
        return "\n".join(lines)

    def query(
        self,
        question: str,
        top_k: int = 5,
        min_score: float = 0.2,
        summarize: bool = False,
        stream: bool = False,
        use_history: bool = True,
        history_window: int = 4,
    ) -> Dict[str, Any]:
        history_context = self._format_history(history_window=history_window) if use_history else ""
        retrieval_query = question
        if history_context:
            retrieval_query = (
                f"Conversation history:\n{history_context}\n\nCurrent user question:\n{question}"
            )

        results = self.retriever.retrieve(
            retrieval_query, top_k=top_k, score_threshold=min_score
        )

        if not results:
            answer = "No relevant answer found"
            sources: List[Dict[str, Any]] = []
            context = ""
        else:
            context = "\n\n".join([doc["content"] for doc in results])
            sources = [
                {
                    "source": doc["metadata"].get(
                        "source_file", doc["metadata"].get("source", "unknown")
                    ),
                    "page": doc["metadata"].get("page", "unknown"),
                    "score": doc["similarity_score"],
                    "preview": doc["content"][:400] + "...",
                }
                for doc in results
            ]

            prompt = f"""You are a helpful RAG assistant.
Use the conversation history and retrieved context to answer the latest question in depth.
Write a comprehensive response with:
1) Direct answer
2) Detailed explanation
3) Key concepts
4) Practical example(s)
5) Caveats or limitations when relevant
If the answer is not present in context, explicitly say so.

Conversation history:
{history_context if history_context else "No prior history."}

Retrieved context:
{context}

Latest question:
{question}

Answer:
"""
            response = self.llm.invoke(prompt)
            answer = _extract_response_text(response)

        if not results:
            # Fall back to conversational response style even when retrieval misses.
            fallback_prompt = f"""You are a helpful assistant.
Conversation history:
{history_context if history_context else "No prior history."}

User question:
{question}

Give a detailed answer. If you are unsure, say so explicitly.
"""
            response = self.llm.invoke(fallback_prompt)
            answer = _extract_response_text(response)

        if stream and answer:
            print("Streaming answer:")
            for i in range(0, len(answer), 80):
                print(answer[i : i + 80], end="", flush=True)
                time.sleep(0.05)
            print("\n")

        citations = [
            f"[{i + 1}] {src['source']} (page {src['page']})"
            for i, src in enumerate(sources)
        ]
        answer_with_citations = (
            f"{answer}\n\nCitations:\n{' '.join(citations)}" if citations else answer
        )

        summary = None
        if summarize and answer:
            summarize_prompt = f"""Use the following answer to summarize concisely.
Answer:
{answer}

Summary:
"""
            summarize_response = self.llm.invoke(summarize_prompt)
            summary = _extract_response_text(summarize_response)

        record = {
            "question": question,
            "answer": answer,
            "sources": sources,
            "summary": summary,
        }
        self.history.append(record)

        return {
            "question": question,
            "answer": answer_with_citations,
            "sources": sources,
            "summary": summary,
            "history": self.history,
        }
