"""
LangChain adapter for the health-report tagger.

The core pipeline (pipeline.py, retrieval.py, llm.py, verify.py) is deliberately
framework-free. This package only *wraps* it as standard LangChain components,
without changing its behaviour:

  * HealthReportRetriever  — BaseRetriever over retrieval.Retriever.search()
  * FindingsRetriever      — BaseRetriever returning the report-driven context
                             (retrieval.Retriever.retrieve()) for given findings
  * analyze_health_report  — tool: PDF -> findings / conditions / advice / verification
  * lookup_reference_range — tool: lab name -> canonical item + public range + source

Requires `pip install -r requirements-langchain.txt`. Import path is
`integrations.langchain` (run from the repo root).
"""

from .retriever import FindingsRetriever, HealthReportRetriever, chunk_to_document, documents_to_context
from .tools import TOOLS, analyze_health_report, lookup_reference_range, summarize_result

__all__ = [
    "HealthReportRetriever", "FindingsRetriever", "chunk_to_document", "documents_to_context",
    "analyze_health_report", "lookup_reference_range", "summarize_result", "TOOLS",
]
