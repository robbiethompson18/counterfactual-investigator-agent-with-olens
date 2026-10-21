FROM python:3.12-slim
RUN pip install --no-cache-dir pytest numpy
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
