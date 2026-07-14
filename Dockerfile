FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    anthropic \
    openai \
    google-genai \
    python-dotenv \
    requests

COPY scripts/ scripts/

CMD ["python", "courage_eval.py"]