# A container image for running our training code anywhere Docker
# runs — not just Azure. CI builds this on every push as a smoke test
# (if it fails to build, something's broken before it ever reaches
# Azure). We'll extend this in a later phase for serving predictions.
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENTRYPOINT ["python", "-m", "src.cloud.train_job_entry"]
