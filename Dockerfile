
# Microsoft ka official Playwright Python image use karenge
# Isme Python aur Browsers pehle se installed hote hain
FROM mcr.microsoft.com/playwright/python:v1.44.0-jammy

# Working directory set karo
WORKDIR /app

# Files copy karo
COPY requirements.txt .
COPY main.py .

# Python libraries install karo
RUN pip install --no-cache-dir -r requirements.txt

# Playwright ke browsers confirm karo (waise base image m hote hain par safe side k liye)
RUN playwright install chromium
RUN playwright install-deps

# Server start karo
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
