"""
Tool-calling demo: ChatOllama (qwen2.5:7b) + the two health-report tools.

    .venv/Scripts/python examples/langchain_agent_demo.py [question]

A minimal, explicit agent loop (model -> tool calls -> ToolMessages -> model)
so every step is visible. The tools are deterministic wrappers around the
framework-free core; the PDF is synthetic (eval/synth/samples).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langchain_ollama import ChatOllama  # noqa: E402

import config  # noqa: E402
from integrations.langchain import TOOLS  # noqa: E402

PDF = (ROOT / "eval" / "synth" / "samples" / "syn_011.pdf").as_posix()
SYSTEM = ("你是健檢報告助理。回答前先用工具取得事實：分析報告用 analyze_health_report"
          "（只問異常時可設 run_llm=false），查參考範圍用 lookup_reference_range。"
          "只根據工具結果以繁體中文回答，並附上資料來源網址。這不是醫療診斷。")


def run(question: str, max_steps: int = 4) -> str:
    llm = ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL,
                     temperature=0, num_ctx=config.LLM_NUM_CTX)
    llm_with_tools = llm.bind_tools(TOOLS)
    tools = {t.name: t for t in TOOLS}
    messages = [SystemMessage(SYSTEM), HumanMessage(question)]
    for _ in range(max_steps):
        ai = llm_with_tools.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            return ai.content
        for call in ai.tool_calls:
            print(f"-> tool {call['name']}({json.dumps(call['args'], ensure_ascii=False)})")
            messages.append(tools[call["name"]].invoke(call))  # ToolCall in -> ToolMessage out
    return llm.invoke(messages).content  # step budget exhausted: answer without tools


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or f"報告檔案：{PDF}\n這份報告有哪些異常？LDL 的參考範圍是多少？"
    print(run(q))
