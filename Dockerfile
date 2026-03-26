FROM python:3.11-slim

LABEL maintainer="HLA-LA Team"
LABEL version="2.0.0"
LABEL description="HLA-LA: HLA typing from NGS data"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    bwa \
    samtools \
    minimap2 \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install HLA-LA Python package
WORKDIR /opt/hlala
COPY pyproject.toml .
COPY hlala/ hlala/
RUN pip install --no-cache-dir .

# Default entrypoint
ENTRYPOINT ["hlala"]
