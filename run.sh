#!/usr/bin/env bash
set -e

echo "=== Health Report Tagger Setup ==="

# 1. Python venv
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate
echo "Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# 2. Check Ollama
echo ""
echo "=== Checking Ollama ==="
if ! command -v ollama &>/dev/null; then
  echo "Ollama not found. Installing..."
  curl -fsSL https://ollama.com/install.sh | sh
else
  echo "Ollama already installed: $(ollama --version)"
fi

# 3. Start Ollama if not running
if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
  echo "Starting Ollama server in background..."
  ollama serve &>/tmp/ollama.log &
  sleep 3
fi

# 4. Pull required models
echo ""
echo "=== Pulling models ==="
echo "Pulling qwen2.5:7b (LLM)..."
ollama pull qwen2.5:7b
echo "Pulling qwen3-embedding:0.6b (embeddings for retrieval)..."
ollama pull qwen3-embedding:0.6b
echo "Pulling glm-ocr (optional, for scanned PDFs)..."
ollama pull glm-ocr || echo "glm-ocr unavailable: scanned PDFs will not be OCR-ed"

# 5. Launch the Gradio UI (local only: http://127.0.0.1:7860)
echo ""
echo "=== Launching the UI ==="
python ui.py --host 127.0.0.1 --port 7860
