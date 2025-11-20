# Use the Playwright base image that contains matching browsers
FROM mcr.microsoft.com/playwright/python:v1.44.0-jammy

WORKDIR /app

COPY requirements.txt .
COPY main.py .

# Install python deps. Note: playwright is NOT in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--proxy-headers"]
