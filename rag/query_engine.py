from __future__ import annotations

import argparse
from functools import lru_cache

from llama_index.core import get_response_synthesizer

from rag.citation_builder import format_citations, prepare_citation_context
from rag.config import DEFAULT_RESPONSE_MODE
from rag.models import QueryResult
from rag.retriever import get_hybrid_retriever
from rag.settings import configure_settings, get_llm


class TaxLawQueryEngine:
    def __init__(self, retriever=None, llm=None) -> None:
        configure_settings()
        self._retriever = retriever or get_hybrid_retriever()
        self._llm = llm or get_llm()
        self._response_synthesizer = get_response_synthesizer(
            llm=self._llm,
            response_mode=DEFAULT_RESPONSE_MODE,
        )

    def query(self, question: str) -> QueryResult:
        source_nodes = self._retriever.retrieve(question)
        if not source_nodes:
            return QueryResult(
                question=question,
                answer=(
                    "I could not retrieve relevant tax-law material for that question. "
                    "Please ingest source documents first or broaden the query."
                ),
                citations=[],
                retrieved_chunks=0,
            )

        citable_nodes, citations = prepare_citation_context(source_nodes)
        response = self._response_synthesizer.synthesize(question, nodes=citable_nodes)
        answer = str(response).strip()
        sources_block = format_citations(citations)
        if sources_block and sources_block not in answer:
            answer = f"{answer}\n\n{sources_block}"

        return QueryResult(
            question=question,
            answer=answer,
            citations=citations,
            retrieved_chunks=len(citations),
        )


@lru_cache(maxsize=1)
def get_query_engine() -> TaxLawQueryEngine:
    return TaxLawQueryEngine()


def answer_question(question: str) -> QueryResult:
    return get_query_engine().query(question)


def reset_query_engine_cache() -> None:
    get_query_engine.cache_clear()


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the Indian tax-law RAG engine.")
    parser.add_argument("question", help="Question to ask against the ingested corpus.")
    args = parser.parse_args()

    result = answer_question(args.question)
    print(result.answer)


if __name__ == "__main__":
    main()
