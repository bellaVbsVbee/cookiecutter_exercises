# Base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY models/ models/
COPY reports/ reports/
COPY data/ data/
COPY pyproject.toml ./
COPY uv.lock ./
WORKDIR /
RUN uv sync --locked --no-cache
ENTRYPOINT ["uv", "run", "src/s2m6_ccex/evaluate.py"]
