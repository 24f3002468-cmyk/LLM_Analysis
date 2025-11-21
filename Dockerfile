# Use the official Playwright image that includes Python3.10 + Chromium/Firefox/WebKit
FROM mcr.microsoft.com/playwright/python:latest

WORKDIR /app

COPY requirements.txt .
COPY main.py .

# Install dependencies (Playwright is already installed in this base image)
RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=80

# Expose the port Render expects
EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--proxy-headers"]
