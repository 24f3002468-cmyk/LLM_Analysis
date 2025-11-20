# Use Playwright official image (contains Python + browsers)
FROM mcr.microsoft.com/playwright/python:v1.44.0-jammy

WORKDIR /app

# Copy files
COPY requirements.txt .
COPY main.py .

# Install python deps
RUN pip install --no-cache-dir -r requirements.txt

# Environment port
ENV PORT=80

# Run uvicorn on port 80 (match Render)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--proxy-headers"]
