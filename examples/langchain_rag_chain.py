"""
Minimal LCEL RAG chain over the project's retriever:

    HealthReportRetriever | prompt | ChatOllama | StrOutputParser

    .venv/Scripts/python examples/langchain_rag_chain.py [question]

Uses BM25 over data/kb_sample.json (no embedding model needed); swap in a
dense / hybrid / reranked retrieval.Retriever without touching the chain.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.output_parsers import StrOutputParser  # noqa: E402
from langchain_core.prompts import ChatPromptTemplate  # noqa: E402
from langchain_core.runnables import RunnablePassthrough  # noqa: E402
from langchain_ollama import ChatOllama  # noqa: E402

import config  # noqa: E402
from integrations.langchain import HealthReportRetriever, documents_to_context  # noqa: E402
from kb_index import load_articles, prepare_chunks  # noqa: E402
from retrieval import BM25Index, CheckitemFacts, Retriever  # noqa: E402

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "你是衛教助理。只根據參考資料以繁體中文回答，每個重點句尾標註引用的 [id]；"
               "資料不足時直說。這不是醫療診斷。"),
    ("human", "# 參考資料\n{context}\n\n# 問題\n{question}"),
])


def build_chain(k: int = 4):
    chunks, _ = prepare_chunks(load_articles(str(ROOT / config.KB_JSON_PATH)))
    facts = CheckitemFacts()
    core = Retriever(None, facts=facts, mode="bm25", bm25=BM25Index(chunks, user_terms=list(facts.by_name)))
    retriever = HealthReportRetriever(retriever=core, k=k)
    llm = ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL,
                     temperature=0, num_ctx=config.LLM_NUM_CTX)
    chain = (
        {"context": retriever | documents_to_context, "question": RunnablePassthrough()}
        | PROMPT | llm | StrOutputParser()
    )
    return chain, retriever


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "LDL 偏高該怎麼調整飲食？"
    chain, retriever = build_chain()
    for d in retriever.invoke(q):
        print(f"[{d.metadata['id']}] score={d.metadata['score']:.2f} {d.metadata['url']}")
    print(chain.invoke(q))
