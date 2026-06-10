
python3 -m venv .venv
source .venv/bin/activate

# Install CPU-only PyTorch first, then everything else
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers
pip install -e ".[fastembed,dev,faiss]" --break-system-packages


# or pip install -r src/tracer/benchmarks/embedders/requirements.txt

python3 src/tracer/benchmarks/embedders/embedders.py





