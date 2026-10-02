FROM python:3.13-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE NOTICE ./
COPY cxq_bt/ ./cxq_bt/
RUN python -m pip install --no-cache-dir ".[ui]"

COPY app.py run_tests.py ./
COPY examples/ ./examples/
COPY tests/ ./tests/
COPY docs/ ./docs/
COPY scripts/ ./scripts/

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--browser.gatherUsageStats", "false"]
