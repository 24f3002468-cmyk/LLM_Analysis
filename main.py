# main.py
import os
import json
import time
import io
import re
import base64
import traceback
import asyncio
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
MY_SECRET = os.environ.get("MY_SECRET")  # MUST be set
DEFAULT_BUDGET_SECONDS = int(os.environ.get("QUIZ_BUDGET_SECONDS", "160"))

if not MY_SECRET:
    print("⚠️ WARNING: MY_SECRET not set.")

app = FastAPI(title="LLM Analysis Quiz Agent")

client = None
if AIPIPE_TOKEN:
    client = OpenAI(api_key=AIPIPE_TOKEN, base_url="https://aipipe.org/openrouter/v1")
else:
    print("⚠️ WARNING: NO AI TOKEN SET. LLM DISABLED.")


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
    for l in links or []:
        if not l:
            continue
        low = l.lower()
        if any(k in low for k in ("submit", "answer", "response", "post", "/submit")):
            return urljoin(base_url, l)

    m = re.search(r'<form[^>]+action=["\']([^"\']+)["\']', html, flags=re.I)
    if m:
        return urljoin(base_url, m.group(1))

    m2 = re.search(r'https?://[^\s\'"<>]+/(?:submit|answer|response)[^\s\'"<>]*', html, flags=re.I)
    if m2:
        return m2.group(0)

    for l in links or []:
        if l and l.startswith("http"):
            return l
    return None


# ---------- Scraping ----------
async def scrape_page(page: Page, url: str) -> Tuple[str, str, List[str]]:
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

    try:
        links = await page.evaluate(
            "() => Array.from(document.querySelectorAll('a')).map(a => a.href)"
        )
    except Exception:
        links = []

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
                        try { out.push(atob(m[1])); } catch {}
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

    try:
        soup = BeautifulSoup(html, "lxml")
        tables = soup.find_all("table")
        for tbl in tables:
            try:
                df = pd.read_html(str(tbl))[0]
            except Exception:
                continue
            for col in df.columns:
                name = str(col).lower().strip()
                if name in ("value", "amount", "cost", "price", "sum", "total"):
                    colnums = pd.to_numeric(df[col], errors='coerce')
                    if colnums.notna().any():
                        submit_url = find_submit_in_text(html)
                        return float(colnums.sum()), (urljoin(base_url, submit_url) if submit_url else None)

            for col in df.columns:
                colnums = pd.to_numeric(df[col], errors='coerce')
                if colnums.notna().any():
                    submit_url = find_submit_in_text(html)
                    return float(colnums.sum()), (urljoin(base_url, submit_url) if submit_url else None)
    except Exception:
        pass

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

    try:
        for l in links or []:
            if l and any(l.lower().endswith(ext) for ext in (".pdf", ".csv", ".xlsx", ".xls")):
                return None, l
    except Exception:
        pass

    final_try = find_submit_in_text(html) or find_submit_in_text(visible_text)
    if final_try:
        return None, urljoin(base_url, final_try)

    return None, None


def is_bad_answer(ans) -> bool:
    if ans is None:
        return True
    if isinstance(ans, bool):
        return False
    if isinstance(ans, (int, float)):
        return False
    if isinstance(ans, str):
        if ans.strip() == "":
            return True
        if MY_SECRET and MY_SECRET in ans:
            return True
        if len(ans) > 1000:
            return True
    return False


# ---------- Main quiz loop ----------
async def process_quiz_loop(start_url: str, email: str, secret: str, total_budget_seconds: int):
    if client is None:
        raise RuntimeError("LLM not available")

    overall_start = time.time()
    current_url = start_url
    visited = set()

    print(f"🚀 Starting Quiz Loop: {current_url}")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()

        while current_url and current_url not in visited:
            visited.add(current_url)

            elapsed_total = time.time() - overall_start
            if elapsed_total >= total_budget_seconds:
                print("⏱ Budget over.")
                break

            print("📍 Processing:", current_url)

            try:
                html, visible_text, links = await scrape_page(page, current_url)

                heuristic_answer, file_or_submit = try_heuristics(html, visible_text, links or [], current_url)

                if heuristic_answer is not None:
                    submission_url = find_submission_url_from_page(html, links or [], current_url)
                    if not submission_url:
                        print("❌ No submission URL.")
                        break

                    payload = {"email": email, "secret": secret, "url": current_url, "answer": heuristic_answer}
                    print("📤 Heuristic submit:", heuristic_answer)

                    res = await asyncio.to_thread(safe_post_json, submission_url, payload, 8, 2)
                    print("Res:", redact(str(res)))

                    if isinstance(res, dict) and res.get("correct") and res.get("url"):
                        current_url = res["url"]
                        continue
                    break

                if file_or_submit and any(file_or_submit.lower().endswith(ext) for ext in (".pdf", ".csv", ".xlsx", ".xls")):
                    try:
                        r = await asyncio.to_thread(requests.get, file_or_submit, 12)
                        if r.status_code == 200 and file_or_submit.endswith(".pdf") and PdfReader:
                            reader = PdfReader(io.BytesIO(r.content))
                            text = ""
                            for pg in reader.pages:
                                try: text += pg.extract_text() or ""
                                except: pass

                            nums = re.findall(r'-?\d+\.?\d*', text)
                            if nums:
                                total = sum(float(n) for n in nums)
                                submission_url = find_submission_url_from_page(html, links or [], current_url)
                                if submission_url:
                                    payload = {"email": email, "secret": secret, "url": current_url, "answer": total}
                                    res = await asyncio.to_thread(safe_post_json, submission_url, payload, 8, 2)
                                    print("PDF res:", redact(str(res)))
                                    if isinstance(res, dict) and res.get("correct") and res.get("url"):
                                        current_url = res["url"]
                                        continue
                                break
                    except Exception:
                        pass

                elapsed_total = time.time() - overall_start
                if total_budget_seconds - elapsed_total < 10:
                    print("⏱ Not enough time for LLM.")
                    break

                prompt = (
                    "You are an expert data analyst. Output ONLY valid JSON.\n\n"
                    "PAGE CONTENT:\n"
                    + (visible_text[:12000] or "")
                    + "\n\nLINKS:"
                    + json.dumps(links[:40])
                    + "\n\nFORMAT:\n"
                    '{"answer": <value>, "submission_url": "<url_or_null>"}'
                )

                try:
                    resp = client.chat.completions.create(
                        model="openai/gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=300,
                        temperature=0
                    )
                    raw = resp.choices[0].message.content.strip()
                    raw = re.sub(r'^```(?:json|text)?\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()
                    print("LLM raw:", redact(raw[:500]))

                    try:
                        data = json.loads(raw)
                    except:
                        data = {"answer": None, "submission_url": None}
                except Exception as e:
                    print("LLM fail:", e)
                    data = {"answer": None, "submission_url": None}

                answer = data.get("answer")
                submission_url = data.get("submission_url") or find_submission_url_from_page(html, links or [], current_url)

                if is_bad_answer(answer) or not submission_url:
                    print("Invalid LLM answer.")
                    break

                payload = {"email": email, "secret": secret, "url": current_url, "answer": answer}
                print("📤 LLM submit:", answer)

                res = await asyncio.to_thread(safe_post_json, submission_url, payload, 8, 2)
                print("LLM res:", redact(str(res)))

                if isinstance(res, dict) and res.get("correct") and res.get("url"):
                    current_url = res["url"]
                    continue
                break

            except Exception as e:
                print("🔥 Error:", e)
                traceback.print_exc()
                break

        await context.close()
        await browser.close()

    print("⏹️ Quiz Ended")


# ---------- Endpoint ----------
@app.post("/quiz")
async def receive_task(task: QuizTask):
    if not MY_SECRET:
        raise HTTPException(status_code=503, detail="Server missing MY_SECRET")
    if task.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    try:
        await process_quiz_loop(task.url, task.email, task.secret, DEFAULT_BUDGET_SECONDS)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": "Processed"}


@app.get("/")
def health():
    return {"status": "ok", "service": "llm-analysis", "version": "1.1"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 80)))
