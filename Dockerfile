FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    anthropic \
    openai \
    google-genai \
    python-dotenv

COPY scripts/courage_eval.py .

CMD ["python", "courage_eval.py"]