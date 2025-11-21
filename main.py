# main.py
import os
import json
import time
import io
import re
import base64
import traceback
from typing import List, Tuple, Optional
from urllib.parse import urljoin, urlparse

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Playwright & other optional libs
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
# Total budget for the whole quiz loop (seconds)
DEFAULT_BUDGET_SECONDS = 110

if not MY_SECRET:
    print("⚠️ WARNING: MY_SECRET not set. Use a dev secret for testing.")

app = FastAPI(title="LLM Analysis Quiz Agent")

client = None
if AIPIPE_TOKEN:
    client = OpenAI(api_key=AIPIPE_TOKEN, base_url="https://aipipe.org/openrouter/v1")
else:
    print("⚠️ WARNING: AIPIPE_TOKEN/OPENAI_API_KEY not set. LLM fallback will be disabled.")

# ---------- Pydantic model ----------
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

def safe_post_json(url: str, payload: dict, timeout=10, retries=3):
    """POST JSON with retries and return parsed JSON or raise."""
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
                # Non-JSON OK responses
                return {"raw_text": r.text, "status_code": r.status_code}
        except requests.RequestException as e:
            last_exc = e
            if i == retries - 1:
                raise
            time.sleep(0.5 * (2 ** i))
    raise last_exc

def find_submission_url_from_page(html: str, links: List[str], base_url: str) -> Optional[str]:
    """Heuristics to locate a submission URL."""
    # 1) direct links with keywords
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
    """Navigate page and return (html, visible_text, links). Also attempts to decode atob() found in scripts via page.evaluate."""
    # Try networkidle navigation with reasonable timeout, fallback to domcontentloaded + small wait
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

    links = await page.evaluate("() => Array.from(document.querySelectorAll('a')).map(a => a.href)")

    # Client-side atob extraction: capture decoded blobs from script tags if present
    try:
        decoded_blobs = await page.evaluate("""
            () => {
                const out = [];
                const scripts = Array.from(document.querySelectorAll('script'));
                for (const s of scripts) {
                    const txt = s.textContent || '';
                    // find atob(`...`) or atob("...") or atob('...')
                    const re = /atob\\((?:`|["'])([^`"']+)(?:`|["'])\\)/g;
                    let m;
                    while ((m = re.exec(txt)) !== null) {
                        try {
                            const dec = atob(m[1]);
                            out.push(dec);
                        } catch (e) {
                            // ignore
                        }
                    }
                }
                return out;
            }
        """)
        if decoded_blobs:
            # append to html and visible_text so heuristics and LLM see it
            joined = "\n\n".join(decoded_blobs)
            html += "\n\n<!-- DECODED_BLOBS -->\n" + joined
            visible_text += "\n\n" + joined
    except Exception:
        # non-fatal
        pass

    return html, visible_text, links

# ---------- Heuristics ----------
def try_heuristics(html: str, visible_text: str, links: List[str], base_url: str):
    """
    Fast deterministic attempts to extract an answer or a file link or a submit URL.
    Returns (answer_or_None, submission_url_or_filelink_or_None)
    """
    def find_submit_in_text(text: str) -> Optional[str]:
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

    # 1) look for JSON inside decoded blobs or inline scripts
    try:
        # try to find JSON object inside html (common for demo using atob)
        jm = re.search(r'(\{[^{}]{10,}\})', html, flags=re.S)
        if jm:
            try:
                obj = json.loads(jm.group(1))
                # if JSON contains an explicit answer field
                if isinstance(obj, dict) and "answer" in obj:
                    sub = None
                    for key in ("submission_url", "submit", "submit_url", "url", "action"):
                        if key in obj and isinstance(obj[key], str) and obj[key].strip():
                            sub = obj[key]; break
                    # fallback: search text for submit link
                    if not sub:
                        sub = find_submit_in_text(html)
                    return obj.get("answer"), (urljoin(base_url, sub) if sub else None)
            except Exception:
                pass
    except Exception:
        pass

    # 2) parse HTML tables - find numeric columns to sum
    try:
        soup = BeautifulSoup(html, "lxml")
        tables = soup.find_all("table")
        for tbl in tables:
            try:
                df = pd.read_html(str(tbl))[0]
            except Exception:
                continue
            # prioritize likely columns
            for col in df.columns:
                name = str(col).strip().lower()
                if name in ("value", "amount", "cost", "price", "sum", "total"):
                    colnums = pd.to_numeric(df[col], errors='coerce')
                    if colnums.notna().any():
                        submit_url = find_submit_in_text(html)
                        return float(colnums.sum(skipna=True)), (urljoin(base_url, submit_url) if submit_url else None)
            # fallback: sum numeric column if any
            for col in df.columns:
                colnums = pd.to_numeric(df[col], errors='coerce')
                if colnums.notna().any():
                    submit_url = find_submit_in_text(html)
                    return float(colnums.sum(skipna=True)), (urljoin(base_url, submit_url) if submit_url else None)
    except Exception:
        pass

    # 3) visible_text numeric extraction: sum small number of numeric tokens
    try:
        nums = re.findall(r'-?\d+\.?\d*', visible_text)
        if nums:
            # if more than one small number, sum them; else return first
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

    # 4) downloadable files (pdf, csv, xlsx) - return the file link for later processing
    try:
        for l in links or []:
            if l and any(l.lower().endswith(ext) for ext in (".pdf", ".csv", ".xlsx", ".xls")):
                # we return file link as submission_url slot to signal further handling
                return None, l
    except Exception:
        pass

    # 5) try to find a submit url present in visible HTML
    try:
        final_try = find_submit_in_text(html) or find_submit_in_text(visible_text)
        if final_try:
            return None, urljoin(base_url, final_try)
    except Exception:
        pass

    return None, None

# ---------- Answer validation ----------
def is_bad_answer(ans) -> bool:
    """Permissive but safe checks. Reject only None, empty string, overly large strings, or strings containing secret."""
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
        # Launch with safe args - keep headless and minimal flags
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

                # 1) Heuristics-first
                heuristic_answer, file_or_submit = try_heuristics(html, visible_text, links or [], current_url)
                if heuristic_answer is not None:
                    # heuristic found numeric/explicit answer
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

                # 2) If heuristics returned a file link (e.g., PDF), try to download and parse quickly
                if file_or_submit:
                    file_link = file_or_submit
                    # if file_or_submit is an absolute submit URL, handle that later; only act if file ext
                    if any(file_link.lower().endswith(ext) for ext in (".pdf", ".csv", ".xlsx", ".xls")):
                        try:
                            r = requests.get(file_link, timeout=12)
                            if r.status_code == 200:
                                if file_link.lower().endswith(".pdf") and PdfReader:
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
                                # add CSV/XLSX parsing if needed (pandas) - omitted for brevity; can be added similarly
                        except Exception:
                            pass

                # 3) LLM fallback: only if enough time remains (leave margin for submission and next steps)
                elapsed_total = time.time() - overall_start
                time_left = total_budget_seconds - elapsed_total
                if time_left < 10:
                    print(f"⏱️ Not enough time left for LLM call (time_left={time_left:.1f}s). Skipping.")
                    break

                if client is None:
                    print("⚠️ No AI client configured; skipping LLM fallback.")
                    break

                # Build a strict, small prompt without injecting secrets
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

                # Make the LLM call (constrained)
                try:
                    resp = client.chat.completions.create(
                        model="openai/gpt-4o-mini",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=200,
                        temperature=0.0
                    )
                    raw = resp.choices[0].message.content.strip()
                    raw = re.sub(r'^```(?:json|text)?\\s*|\\s*```$', '', raw, flags=re.MULTILINE).strip()
                    # Log the raw LLM output (redacted)
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
                    # If LLM gave a reason, log it (redacted)
                    if isinstance(data, dict) and data.get("reason"):
                        print("LLM reason:", redact(str(data.get("reason"))))
                    break

                # Submit LLM answer
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
    # Validate secret quickly
    if not MY_SECRET:
        raise HTTPException(status_code=503, detail="Server not configured with MY_SECRET")
    if task.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
    # Process synchronously so logs appear in Render output and Render doesn't kill background tasks.
    try:
        await process_quiz_loop(task.url, task.email, task.secret, total_budget_seconds=DEFAULT_BUDGET_SECONDS)
    except Exception as e:
        # Do not leak sensitive info
        raise HTTPException(status_code=500, detail=str(e))
    return {"message": "Processed"}

@app.get("/")
def health():
    return {"status": "ok", "service": "llm-analysis", "version": "1.0"}

# ---------- Run ----------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", 80)))
