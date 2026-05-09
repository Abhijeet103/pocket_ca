"""Core package for the Indian tax-law RAG assistant."""

from __future__ import annotations


__all__ = [
    "TaxChatbot",
    "TaxLawQueryEngine",
    "calculate_tax",
    "compare_old_vs_new_regime",
    "explain_tax_breakdown",
    "get_query_engine",
    "list_missing_information",
    "suggest_applicable_deductions",
]


def __getattr__(name: str):
    if name == "TaxChatbot":
        from rag.chatbot import TaxChatbot

        return TaxChatbot

    if name in {"TaxLawQueryEngine", "get_query_engine"}:
        from rag.query_engine import TaxLawQueryEngine, get_query_engine

        exports = {
            "TaxLawQueryEngine": TaxLawQueryEngine,
            "get_query_engine": get_query_engine,
        }
        return exports[name]

    if name in {
        "calculate_tax",
        "compare_old_vs_new_regime",
        "explain_tax_breakdown",
        "list_missing_information",
        "suggest_applicable_deductions",
    }:
        from rag.tax_tools import (
            calculate_tax,
            compare_old_vs_new_regime,
            explain_tax_breakdown,
            list_missing_information,
            suggest_applicable_deductions,
        )

        exports = {
            "calculate_tax": calculate_tax,
            "compare_old_vs_new_regime": compare_old_vs_new_regime,
            "explain_tax_breakdown": explain_tax_breakdown,
            "list_missing_information": list_missing_information,
            "suggest_applicable_deductions": suggest_applicable_deductions,
        }
        return exports[name]

    raise AttributeError(f"module 'rag' has no attribute {name!r}")
