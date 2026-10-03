"""Retrieval runs fully locally, so these tests check the real index - no LLM involved."""

import pytest

from agents import make_search_tool
from evaluate import EVAL_CASES

ANSWERABLE = [(q, sources) for q, sources, _ in EVAL_CASES if sources]


def test_all_pdfs_indexed(rag):
    sources = {doc["metadata"]["source"] for doc in rag.vector_store.store.values()}
    assert sources == {"employee_handbook.pdf", "project_kestrel_spec.pdf",
                       "q3_2026_sales_report.pdf", "incident_postmortem_inc0917.pdf"}


@pytest.mark.parametrize("question,expected_sources", ANSWERABLE)
def test_retriever_finds_expected_pdf(rag, question, expected_sources):
    retrieved = {d.metadata["source"] for d in rag.retriever.invoke(question)}
    assert set(expected_sources) <= retrieved


def test_scoped_search_tool_only_sees_its_pdfs(rag):
    search = make_search_tool(rag.vector_store, "search_hr_documents", "HR docs",
                              {"employee_handbook.pdf"})
    # A sales question must still only return handbook chunks.
    result = search.invoke({"query": "What was total revenue in Q3 2026?"})
    assert "[employee_handbook.pdf]" in result
    assert "q3_2026_sales_report.pdf" not in result
