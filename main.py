# main.py
import os
import json
import time
import io
import re
import base64
import traceback
from typing import List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Playwright & OpenAI client
from playwright.async_api import async_playwright, Page
from openai import OpenAI

# Optional helpers
from bs4 import BeautifulSoup
import pandas as pd
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

# ---------- CONFIG ----------
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN") or os.environ.get("OPENAI_API_KEY")
MY_SECRET = os.environ.get("MY_SECRET")  # MUST be set in environment
DEFAULT_BUDGET_SECONDS = 110  # two-minute-ish budget you requested

if not MY_SECRET:
    print("⚠️ WARNING: MY_SECRET not set. Use a dev secret for testing.")

app = FastAPI(title="LLM Analysis Quiz Agent")

client = None
if AIPIPE_TOKEN:
    client = OpenAI(api_key=AIPIPE_TOKEN, base_url="https://aipipe.org/openrouter/v1")
else:
    print("⚠️ WARNING: AIPIPE_TOKEN/OPENAI_API_KEY not set. LLM fallback will be disabled.")

class QuizTask(BaseModel):
    email: str
    secret: str
    url: str

# ---------- Utilities ----------
def redact(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    if MY_SECRET:
        return s.replace(MY_SECRET, "<REDACTED>")
    return s

def safe_post_json(url: str, payload: dict, timeout: int = 10, retries: int = 3):
    if urlparse(url).scheme not in ("http", "https"):
        raise ValueError("Invalid URL scheme")
    last_exc = None
    for i in range(retries):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            try:
                return r.json()
            except ValueError:
                return {"raw_text": r.text, "status_code": r.status_code}
        except requests.RequestException as e:
            last_exc = e
            if i == retries - 1:
                raise
            time.sleep(0.5 * (2 ** i))
    raise last_exc

def find_submission_url_from_page(html: str, links: List[str], base_url: str) -> Optional[str]:
    # 1) direct links with submit-like keywords
    for l in links or []:
        if not l:
            continue
        low = l.lower()
        if any(k in low for k in ("submit", "answer", "response", "post", "/submit")):
            return urljoin(base_url, l)
    # 2) form action
    m = re.search(r'<form[^>]+action=["\']([^"\']+)["\']', html, flags=re.I)
    if m:
        return urljoin(base_url, m.group(1))
    # 3) script-embedded explicit urls
    m2 = re.search(r'https?://[^\s\'"<>]+/(?:submit|answer|response)[^\s\'"<>]*', html, flags=re.I)
    if m2:
        return m2.group(0)
    # 4) fallback: first absolute link
    for l in links or []:
        if l and l.startswith("http"):
            return l
    return None

# ---------- Scraping ----------
async def scrape_page(page: Page, url: str) -> Tuple[str, str, List[str]]:
    # Navigate with fallback strategies
    try:
        await page.goto(url, wait_until="networkidle", timeout=25000)
    except Exception:
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=20000)
            await page.wait_for_timeout(800)
        except Exception:
            pass

    try:
        await page.wait_for_selector("body", timeout=3000)
    except Exception:
        pass

    html = await page.content()
    visible_text = ""
    try:
        sel = await page.query_selector("body")
        if sel:
            visible_text = await sel.inner_text()
    except Exception:
        visible_text = ""

    links = []
    try:
        links = await page.evaluate("() => Array.from(document.querySelectorAll('a')).map(a => a.href)")
    except Exception:
        links = []

    # Extract any atob(...) payloads client-side and append them
    try:
        decoded_blobs = await page.evaluate("""
            () => {
                const out = [];
                const scripts = Array.from(document.querySelectorAll('script'));
                const re = /atob\\((?:`|["'])([^`"']+)(?:`|["'])\\)/g;
                for (const s of scripts) {
                    const txt = s.textContent || '';
                    let m;
                    while ((m = re.exec(txt)) !== null) {
                        try {
                            out.push(atob(m[1]));
                        } catch (e) {
                            // ignore decode errors
                        }
                    }
                }
                return out;
            }
        """)
        if decoded_blobs:
            joined = "\n\n".join(decoded_blobs)
            html += "\n\n<!-- DECODED_BLOBS -->\n" + joined
            visible_text += "\n\n" + joined
    except Exception:
        pass

    return html, visible_text, links

# ---------- Heuristics ----------
def try_heuristics(html: str, visible_text: str, links: List[str], base_url: str):
    def find_submit_in_text(text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        m = re.search(r'https?://[^\s\'"<>]*?/submit[^\s\'"<>]*', text, flags=re.I)
        if m:
            return m.group(0)
        m2 = re.search(r'https?://[^\s\'"<>]*?(submit|answer|response)[^\s\'"<>]*', text, flags=re.I)
        if m2:
            return m2.group(0)
        m3 = re.search(r'https?://[^\s\'"<>]+', text)
        if m3:
            return m3.group(0)
        return None

    # 1) JSON blob in page (common when using atob)
    try:
        jm = re.search(r'(\{[^{}]{10,}\})', html, flags=re.S)
        if jm:
            try:
                obj = json.loads(jm.group(1))
                if isinstance(obj, dict) and "answer" in obj:
                    sub = None
                    for key in ("submission_url", "submit", "submit_url", "url", "action"):
                        if key in obj and isinstance(obj[key], str) and obj[key].strip():
                            sub = obj[key]; break
                    if not sub:
                        sub = find_submit_in_text(html)
                    return obj.get("answer"), (urljoin(base_url, sub) if sub else None)
            except Exception:
                pass
    except Exception:
        pass

    # 2) HTML tables via pandas
    try:
        soup = BeautifulSoup(html, "lxml")
        tables = soup.find_all("table")
        for tbl in tables:
            try:
                df = pd.read_html(str(tbl))[0]
            except Exception:
                continue
            for col in df.columns:
                name = str(col).strip().lower()
                if name in ("value", "amount", "cost", "price", "sum", "total"):
                    colnums = pd.to_numeric(df[col], errors='coerce')
                    if colnums.notna().any():
                        submit_url = find_submit_in_text(html)
                        return float(colnums.sum(skipna=True)), (urljoin(base_url, submit_url) if submit_url else None)
            # fallback numeric column
            for col in df.columns:
                colnums = pd.to_numeric(df[col], errors='coerce')
                if colnums.notna().any():
                    submit_url = find_submit_in_text(html)
                    return float(colnums.sum(skipna=True)), (urljoin(base_url, submit_url) if submit_url else None)
    except Exception:
        pass

    # 3) visible text numbers
    try:
        nums = re.findall(r'-?\d+\.?\d*', visible_text)
        if nums:
            if 1 < len(nums) <= 200:
                vals = [float(n) for n in nums]
                submit_url = find_submit_in_text(visible_text) or find_submit_in_text(html)
                return sum(vals), (urljoin(base_url, submit_url) if submit_url else None)
            else:
                n = nums[0]
                submit_url = find_submit_in_text(visible_text) or find_submit_in_text(html)
                return (float(n) if '.' in n else int(n)), (urljoin(base_url, submit_url) if submit_url else None)
    except Exception:
        pass

    # 4) downloadable files
    try:
        for l in links or []:
            if l and any(l.lower().endswith(ext) for ext in (".pdf", ".csv", ".xlsx", ".xls")):
                return None, l
    except Exception:
        pass

    # 5) final attempt to find explicit submit link
    try:
        final_try = find_submit_in_text(html) or find_submit_in_text(visible_text)
        if final_try:
            return None, urljoin(base_url, final_try)
    except Exception:
        pass

    return None, None

# ---------- Answer validation ----------
def is_bad_answer(ans) -> bool:
    if ans is None:
        return True
    if isinstance(ans, bool):
        return False
    if isinstance(ans, (int, float)):
        return False
    if isinstance(ans, str):
        s = ans.strip()
        if s == "":
            return True
        if MY_SECRET and MY_SECRET in s:
            return True
        if len(s) > 1000:
            return True
    return False

# ---------- Main quiz loop ----------
async def process_quiz_loop(start_url: str, email: str, secret: str, total_budget_seconds: int = DEFAULT_BUDGET_SECONDS):
    if client is None:
        raise RuntimeError("AI client not configured")
    overall_start = time.time()
    current_url = start_url
    visited = set()
    print(f"🚀 Starting Quiz Loop at: {current_url} (budget {total_budget_seconds}s)")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()

        while current_url and current_url not in visited:
            visited.add(current_url)
            elapsed_total = time.time() - overall_start
            if elapsed_total >= total_budget_seconds:
                print(f"⏱️ Time budget exhausted ({elapsed_total:.1f}s). Stopping.")
                break

            print("📍 Processing:", current_url)
            try:
                html, visible_text, links = await scrape_page(page, current_url)
                print(f"✅ Scraped page; found {len(links)} links")

                # Heuristics-first
                heuristic_answer, file_or_submit = try_heuristics(html, visible_text, links or [], current_url)
                if heuristic_answer is not None:
                    submission_url = find_submission_url_from_page(html, links or [], current_url)
                    if not submission_url:
                        print("❌ Submission URL not found after heuristic. Aborting this URL.")
                        break
                    payload = {"email": email, "secret": secret, "url": current_url, "answer": heuristic_answer}
                    print(f"📤 Submitting (heuristic) to {submission_url} with answer: {str(heuristic_answer)[:200]}")
                    try:
                        res = safe_post_json(submission_url, payload, timeout=8, retries=2)
                        print("✅ Submission response:", redact(str(res)))
                        if isinstance(res, dict) and res.get("correct") is True and res.get("url"):
                            current_url = res["url"]
                            continue
                        else:
                            print("🏁 Heuristic submission ended the run.")
                            break
                    except Exception as e:
                        print("❌ Heuristic submission failed:", e)
                        break

                # File handling (PDF quick parse)
                if file_or_submit:
                    file_link = file_or_submit
                    if any(file_link.lower().endswith(ext) for ext in (".pdf", ".csv", ".xlsx", ".xls")):
                        try:
                            r = requests.get(file_link, timeout=12)
                            if r.status_code == 200 and file_link.lower().endswith(".pdf") and PdfReader:
                                try:
                                    reader = PdfReader(io.BytesIO(r.content))
                                    text = ""
                                    for pg in reader.pages:
                                        try:
                                            text += pg.extract_text() or ""
                                        except Exception:
                                            pass
                                    nums = re.findall(r'-?\d+\.?\d*', text)
                                    if nums:
                                        total = sum(float(n) for n in nums)
                                        submission_url = find_submission_url_from_page(html, links or [], current_url)
                                        if submission_url:
                                            payload = {"email": email, "secret": secret, "url": current_url, "answer": total}
                                            print(f"📤 Submitting (pdf heuristic) to {submission_url} with answer: {total}")
                                            res = safe_post_json(submission_url, payload, timeout=8, retries=2)
                                            print("✅ Submission response:", redact(str(res)))
                                            if isinstance(res, dict) and res.get("correct") is True and res.get("url"):
                                                current_url = res["url"]
                                                continue
                                            else:
                                                print("🏁 PDF heuristic ended the run.")
                                                break
                                except Exception:
                                    pass
                        except Exception:
                            pass

                # LLM fallback only if enough time left
                elapsed_total = time.time() - overall_start
                time_left = total_budget_seconds - elapsed_total
                if time_left < 10:
                    print(f"⏱️ Not enough time left for LLM call (time_left={time_left:.1f}s). Skipping.")
                    break

                if client is None:
                    print("⚠️ No AI client configured; skipping LLM fallback.")
                    break

                prompt = (
                    "You are an expert Python data analyst. Output ONLY valid JSON (no markdown, no commentary).\n\n"
                    "PAGE CONTENT (truncated):\n---\n"
                    + (visible_text[:12000] or "")
                    + "\n---\n\nLINKS: "
                    + json.dumps(links[:40])
                    + "\n\nOUTPUT FORMAT (MUST BE EXACT JSON):\n"
                    + '{"answer": <number|string|boolean|null>, "submission_url": "<url_or_relative_or_null>"}'
                    + "\n\nIf you cannot compute a confident answer, return {\"answer\": null, \"submission_url\": null, \"reason\":\"COULD_NOT_COMPUTE\"}."
                )

                try:
                    resp = client.chat.completions.create(
                        model="openai/gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=200,
                        temperature=0.0
                    )
                    raw = resp.choices[0].message.content.strip()
                    raw = re.sub(r'^```(?:json|text)?\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()
                    raw_redacted = redact(raw)
                    print("🧾 Raw LLM output (redacted, first 1000 chars):")
                    print(raw_redacted[:1000])
                    try:
                        data = json.loads(raw)
                    except Exception:
                        print("❌ LLM returned invalid JSON (redacted):", raw_redacted[:300])
                        data = {"answer": None, "submission_url": None}
                except Exception as e:
                    print("❌ LLM call failed:", e)
                    data = {"answer": None, "submission_url": None}

                answer = data.get("answer")
                submission_url = data.get("submission_url") or find_submission_url_from_page(html, links or [], current_url)

                if is_bad_answer(answer) or not submission_url:
                    print("⚠️ LLM answer invalid or unsafe. Skipping submission for this URL.")
                    if isinstance(data, dict) and data.get("reason"):
                        print("LLM reason:", redact(str(data.get("reason"))))
                    break

                payload = {"email": email, "secret": secret, "url": current_url, "answer": answer}
                print(f"📤 Submitting (LLM) to {submission_url} with answer: {str(answer)[:200]}")
                try:
                    res = safe_post_json(submission_url, payload, timeout=8, retries=2)
                    print("✅ Submission response:", redact(str(res)))
                    if isinstance(res, dict) and res.get("correct") is True and res.get("url"):
                        current_url = res["url"]
                        continue
                    else:
                        print("🏁 LLM submission ended the run.")
                        break
                except Exception as e:
                    print("❌ Submission failed:", e)
                    break

            except Exception as e:
                print("🔥 Unexpected error:", e)
                traceback.print_exc()
                break

        await context.close()
        await browser.close()

    print("⏹️ Quiz loop ended. Total elapsed:", round(time.time() - overall_start, 2))

# ---------- Endpoint ----------
@app.post("/quiz")
async def receive_task(task: QuizTask):
    if not MY_SECRET:
        raise HTTPException(status_code=503, detail="Server not configured with MY_SECRET")
    if task.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    try:
        await process_quiz_loop(task.url, task.email, task.secret, total_budget_seconds=DEFAULT_BUDGET_SECONDS)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"message": "Processed"}

@app.get("/")
def health():
    return {"status": "ok", "service": "llm-analysis", "version": "1.0"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 80)))
