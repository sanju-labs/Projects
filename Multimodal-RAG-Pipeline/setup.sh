#!/bin/bash
# ============================================
# Multimodal RAG — One-Command Setup
# ============================================
# Usage:  chmod +x setup.sh && ./setup.sh
# ============================================

set -e

echo ""
echo "🤖 Bing — Multimodal RAG Setup"
echo "==============================="
echo ""

# 1. Check Python
PYTHON=""
for cmd in python3.12 python3; do
    if command -v $cmd &> /dev/null; then
        PYTHON=$cmd
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "❌ Python 3.10+ is required."
    exit 1
fi
echo "✅ Using: $($PYTHON --version)"

# 2. Create venv
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    $PYTHON -m venv .venv
fi
source .venv/bin/activate
echo "✅ Virtual environment active"

# 3. Install deps
echo "📥 Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "✅ All packages installed"

# 4. Setup .env
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ""
    echo "🔑 Created .env — add your OpenAI API key:"
    echo "   nano .env"
    echo ""
else
    echo "✅ .env exists"
fi

# 5. Data dir
mkdir -p data
echo "✅ data/ directory ready"

echo ""
echo "======================================="
echo "🎉 Setup complete!"
echo "======================================="
echo ""
echo "Next steps:"
echo "  1. Add your OpenAI API key to .env"
echo "  2. Drop PDF files into data/"
echo "  3. Run:"
echo ""
echo "     source .venv/bin/activate"
echo "     streamlit run app.py"
echo ""
echo "  The app auto-ingests PDFs on first launch."
echo ""
