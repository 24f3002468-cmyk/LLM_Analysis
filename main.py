import os
import json
import time
import requests
import uvicorn
import traceback
import re
from urllib.parse import urljoin, urlparse

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from playwright.async_api import async_playwright, Page
from openai import OpenAI

# ---------- CONFIG ----------
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN") or os.environ.get("OPENAI_API_KEY")
MY_SECRET = os.environ.get("MY_SECRET")  # do NOT hardcode production secret in repo

if not MY_SECRET:
    print("⚠️ WARNING: MY_SECRET not set. Use a dev secret for testing.")

app = FastAPI()

client = None
if AIPIPE_TOKEN:
    client = OpenAI(api_key=AIPIPE_TOKEN, base_url="https://aipipe.org/openrouter/v1")
else:
    print("⚠️ WARNING: AIPIPE_TOKEN/OPENAI_API_KEY not set. /quiz will return 503 until configured.")

class QuizTask(BaseModel):
    email: str
    secret: str
    url: str

# ---------- UTILITIES ----------
def safe_post_json(url: str, payload: dict, timeout=10, retries=3):
    """POST JSON with retries. Returns parsed JSON or raises."""
    if urlparse(url).scheme not in ("http", "https"):
        raise ValueError("Invalid URL scheme")
    for i in range(retries):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            # try parse json
            try:
                return r.json()
            except ValueError:
                return {"raw_text": r.text, "status_code": r.status_code}
        except requests.RequestException as e:
            if i == retries - 1:
                raise
            time.sleep(0.5 * (2 ** i))

def find_submission_url_from_page(html: str, links: list, base_url: str):
    """Deterministic heuristics to find a submission URL."""
    # 1) explicit links containing keywords
    for l in links:
        if l and any(k in l.lower() for k in ("submit", "answer", "response", "post")):
            return urljoin(base_url, l)
    # 2) form action
    m = re.search(r'<form[^>]+action=["\']([^"\']+)["\']', html, flags=re.I)
    if m:
        return urljoin(base_url, m.group(1))
    # 3) script-embedded url patterns
    m2 = re.search(r'https?://[^\s\'"<>]+/submit[^\s\'"<>]*', html, flags=re.I)
    if m2:
        return m2.group(0)
    # 4) fallback to first absolute link
    for l in links:
        if l and l.startswith("http"):
            return l
    return None

async def scrape_page(page: Page, url: str):
    """Navigate and return (html, visible_text, links)."""
    await page.goto(url, wait_until="networkidle", timeout=45000)
    try:
        await page.wait_for_selector("body", timeout=3000)
    except:
        pass
    html = await page.content()
    visible_text = await page.inner_text("body")
    links = await page.evaluate("() => Array.from(document.querySelectorAll('a')).map(a => a.href)")
    return html, visible_text, links

# ---------- QUIZ PROCESS ----------
async def process_quiz_loop(start_url: str, email: str, secret: str):
    if client is None:
        raise RuntimeError("AI client not configured")
    current_url = start_url
    visited = set()
    print(f"🚀 Starting Quiz Loop at: {current_url}")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()

        while current_url and current_url not in visited:
            visited.add(current_url)
            print("📍 Processing:", current_url)
            try:
                html, visible_text, links = await scrape_page(page, current_url)
                print(f"✅ Scraped page; found {len(links)} links")

                # Ask LLM for JSON ONLY
                prompt = f"""
You are an expert Python data analyst. Use the PAGE CONTENT and LINKS to:
1) Determine the answer required by the page.
2) Provide the submission URL (full or relative).

PAGE CONTENT:
---
{visible_text[:16000]}
---

LINKS:
{links}

RESPONSE FORMAT (MUST BE EXACT VALID JSON ONLY):
{{"answer": <number|string|boolean|null>, "submission_url": "<url_or_relative_or_null>"}}
Output nothing other than the JSON object.
"""
                resp = client.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=[{"role":"user","content": prompt}],
                    max_tokens=800
                )
                raw = resp.choices[0].message.content.strip()
                raw = re.sub(r'^```(?:json|text)?\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()
                try:
                    data = json.loads(raw)
                except Exception as e:
                    print("❌ Model returned invalid JSON. Raw:", raw[:300])
                    data = {"answer": None, "submission_url": None}

                answer = data.get("answer")
                submission_url = data.get("submission_url")

                # Normalize submission_url; fallback heuristics if missing
                if submission_url:
                    submission_url = urljoin(current_url, submission_url)
                else:
                    submission_url = find_submission_url_from_page(html, links or [], current_url)

                if not submission_url:
                    print("❌ Submission URL not found. Aborting this run.")
                    break

                # Prepare payload (per spec)
                payload = {"email": email, "secret": secret, "url": current_url, "answer": answer}
                print(f"📤 Submitting to {submission_url} with answer: {str(answer)[:200]}")

                try:
                    res = safe_post_json(submission_url, payload, timeout=10, retries=3)
                except Exception as e:
                    print("❌ Submission failed:", e)
                    break

                print("✅ Submission response:", res)
                if isinstance(res, dict) and res.get("correct") is True and res.get("url"):
                    current_url = res["url"]
                    # continue to next URL
                else:
                    print("🏁 Quiz finished or incorrect; stopping loop.")
                    break

            except Exception as e:
                print("🔥 Unexpected error:", e)
                traceback.print_exc()
                break

        await context.close()
        await browser.close()

# ---------- ENDPOINT ----------
@app.post("/quiz")
async def receive_task(task: QuizTask):
    # Validate secret quickly
    if not MY_SECRET:
        raise HTTPException(status_code=503, detail="Server not configured with MY_SECRET")
    if task.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    # For testing, process synchronously to ensure Render doesn't kill background tasks.
    try:
        await process_quiz_loop(task.url, task.email, task.secret)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"message": "Processed"}

@app.get("/")
def health():
    return {"status": "ok", "service": "llm-analysis", "version": "1.0"}

# ---------- RUN ----------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 80)))
