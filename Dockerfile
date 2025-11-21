
# Dockerfile
FROM python:3.10-slim

# system deps for Playwright + general tools
RUN apt-get update && apt-get install -y \
    wget gnupg ca-certificates git curl ffmpeg \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libxkbcommon0 \
    libxcomposite1 libxdamage1 libxrandr2 libgbm1 libasound2 libxshmfence1 \
    libpangocairo-1.0-0 libpango-1.0-0 libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers (after playwright pip install)
RUN playwright install chromium

COPY main.py .

EXPOSE 80
ENV PORT=80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
