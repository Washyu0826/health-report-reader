"""
LangChain retrievers over retrieval.Retriever (no behaviour change).

HealthReportRetriever: free-text query -> Retriever.search(query, k)
    (whatever mode the wrapped Retriever uses: dense / bm25 / hybrid / reranked).
FindingsRetriever: abnormal findings -> Retriever.retrieve(findings), i.e. the
    same report-driven, round-robin assembled context the pipeline feeds the LLM
    (fact sheets first). The query string is not used for ranking.

Each hit dict {id, text, score, source, url?, for_finding?} becomes a
Document(page_content=text, id=id, metadata={id, source, url, score, for_finding}).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

import config
from retrieval import Retriever, select_context


def chunk_to_document(hit: Dict[str, Any]) -> Document:
    meta = {
        "id": hit["id"],
        "source": hit.get("source", ""),
        "url": hit.get("url", ""),
        "score": hit.get("score"),
        "for_finding": hit.get("for_finding"),
    }
    return Document(page_content=hit["text"], id=hit["id"], metadata=meta)


def document_to_chunk(doc: Document) -> Dict[str, Any]:
    m = doc.metadata or {}
    return {"id": m.get("id") or doc.id, "text": doc.page_content, "source": m.get("source", ""),
            "url": m.get("url", ""), "score": m.get("score")}


def documents_to_context(docs: List[Document], max_chars: int = config.CONTEXT_MAX_CHARS) -> str:
    """Render Documents exactly like the core prompt does ("[id] (source=...)" blocks
    under the same character budget), so citations in a chain use the same ids."""
    text, _ids = select_context([document_to_chunk(d) for d in docs], max_chars=max_chars)
    return text


class HealthReportRetriever(BaseRetriever):
    """`invoke(query)` -> top-k Documents from `Retriever.search(query, k)`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    retriever: Retriever
    k: int = 4

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return [chunk_to_document(h) for h in self.retriever.search(query, self.k)]

    @classmethod
    def from_findings(cls, retriever: Retriever, findings: List[Dict], **kwargs: Any) -> "FindingsRetriever":
        """Report-driven variant: see FindingsRetriever."""
        return FindingsRetriever(retriever=retriever, findings=findings, **kwargs)


class FindingsRetriever(BaseRetriever):
    """Report-driven context for fixed abnormal findings (Retriever.retrieve).

    `invoke(...)` ignores the query text: the findings *are* the query (one
    query per finding + direct fact-sheet lookup). `last_debug` holds the
    retrieval debug dict (queries, fact_hits, kb_hits) of the latest call.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    retriever: Retriever
    findings: List[Dict] = Field(default_factory=list)
    per_finding_k: int = 2
    max_findings: int = 8
    general_k: int = 3
    last_debug: Optional[Dict] = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        chunks, debug = self.retriever.retrieve(
            self.findings, per_finding_k=self.per_finding_k,
            max_findings=self.max_findings, general_k=self.general_k)
        self.last_debug = debug
        return [chunk_to_document(c) for c in chunks]
