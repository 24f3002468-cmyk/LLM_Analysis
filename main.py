# main.py
import os
import json
import time
import requests
import uvicorn
import traceback
import re
import io
import base64
from urllib.parse import urljoin, urlparse

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from playwright.async_api import async_playwright, Page
from openai import OpenAI

# Optional libs used by heuristics
from bs4 import BeautifulSoup
import pandas as pd
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

# ---------- CONFIG ----------
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN") or os.environ.get("OPENAI_API_KEY")
MY_SECRET = os.environ.get("MY_SECRET")  # must be set in env on Render

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
    if urlparse(url).scheme not in ("http", "https"):
        raise ValueError("Invalid URL scheme")
    for i in range(retries):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            try:
                return r.json()
            except ValueError:
                return {"raw_text": r.text, "status_code": r.status_code}
        except requests.RequestException as e:
            if i == retries - 1:
                raise
            time.sleep(0.5 * (2 ** i))

def find_submission_url_from_page(html: str, links: list, base_url: str):
    for l in links:
        if l and any(k in l.lower() for k in ("submit", "answer", "response", "post")):
            return urljoin(base_url, l)
    m = re.search(r'<form[^>]+action=["\']([^"\']+)["\']', html, flags=re.I)
    if m:
        return urljoin(base_url, m.group(1))
    m2 = re.search(r'https?://[^\s\'"<>]+/submit[^\s\'"<>]*', html, flags=re.I)
    if m2:
        return m2.group(0)
    for l in links:
        if l and l.startswith("http"):
            return l
    return None

async def scrape_page(page: Page, url: str):
    try:
        await page.goto(url, wait_until="networkidle", timeout=8000)
    except Exception:
        try:
            # fallback to domcontentloaded then small wait to let JS run
            await page.goto(url, wait_until="domcontentloaded", timeout=8000)
            await page.wait_for_timeout(500)  # 500ms extra for JS execution
        except:
            pass
    try:
        await page.wait_for_selector("body", timeout=2000)
    except:
        pass
    html = await page.content()
    visible_text = ""
    try:
        sel = await page.query_selector("body")
        if sel:
            visible_text = await sel.inner_text()
    except:
        visible_text = ""
    links = await page.evaluate("() => Array.from(document.querySelectorAll('a')).map(a => a.href)")
    return html, visible_text, links

# ---------- FAST HELPERS ----------
def redact(s: str):
    if not s:
        return s
    if MY_SECRET:
        return s.replace(MY_SECRET, "<REDACTED>")
    return s

def is_bad_answer(ans):
    if ans is None:
        return True
    if isinstance(ans, bool):
        return False
    if isinstance(ans, (int, float)):
        return False
    if isinstance(ans, str):
        s = ans.strip().lower()
        if s in ("", "anything you want", "your secret", "secret", "n/a", "none"):
            return True
        if MY_SECRET and MY_SECRET in ans:
            return True
        if len(s) > 200:
            return True
    return False

def try_heuristics(html, visible_text, links, base_url):
    """Fast deterministic attempts to extract an answer or a file link or a submit URL.
    Returns tuple (answer_or_none, submission_url_or_filelink_or_none).
    """
    # helper to find submit-like URL in a blob of text
    def find_submit_in_text(text):
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

    # 1) atob(`...`) or atob("...") or atob('...') base64 JSON payload detection
    m = re.search(r'atob\((?:`|["\'])([^`"\']+)(?:`|["\'])\)', html)
    if m:
        try:
            decoded = base64.b64decode(m.group(1)).decode('utf-8', errors='ignore')
            # try JSON inside decoded
            jm = re.search(r'(\{.*\})', decoded, flags=re.S)
            if jm:
                try:
                    obj = json.loads(jm.group(1))
                    # if JSON contains an explicit answer field
                    if isinstance(obj, dict) and "answer" in obj:
                        submit_from_json = None
                        for key in ("submission_url", "submit", "submit_url", "url", "action"):
                            if key in obj and isinstance(obj[key], str) and obj[key].strip():
                                submit_from_json = obj[key]
                                break
                        if submit_from_json:
                            # normalize later using urljoin
                            return obj["answer"], submit_from_json
                        # no explicit submit url in json, but maybe present as full url in decoded blob
                        found = find_submit_in_text(decoded)
                        return obj["answer"], found
                except:
                    pass
            # fallback: if decoded contains numbers, sum them
            nums = re.findall(r'-?\d+\.?\d*', decoded)
            if nums:
                vals = [float(n) for n in nums]
                found = find_submit_in_text(decoded)
                return sum(vals), found
        except Exception:
            pass

    # 2) HTML tables - quick parse and sum of likely numeric column
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
                        return float(colnums.sum(skipna=True)), submit_url
            for col in df.columns:
                colnums = pd.to_numeric(df[col], errors='coerce')
                if colnums.notna().any():
                    submit_url = find_submit_in_text(html)
                    return float(colnums.sum(skipna=True)), submit_url
    except Exception:
        pass

    # 3) direct visible_text number extraction
    nums = re.findall(r'-?\d+\.?\d*', visible_text)
    if nums:
        if len(nums) > 1 and len(nums) <= 200:
            vals = [float(n) for n in nums]
            submit_url = find_submit_in_text(visible_text) or find_submit_in_text(html)
            return sum(vals), submit_url
        else:
            n = nums[0]
            submit_url = find_submit_in_text(visible_text) or find_submit_in_text(html)
            return (float(n) if '.' in n else int(n)), submit_url

    # 4) downloadable files
    for l in links:
        if l and any(l.lower().endswith(ext) for ext in (".pdf", ".csv", ".xlsx")):
            script_submit = find_submit_in_text(html)
            return None, script_submit or l

    final_try = find_submit_in_text(html) or find_submit_in_text(visible_text)
    if final_try:
        return None, final_try

    return None, None

# ---------- QUIZ PROCESS (FAST-FIRST, 1-min budget) ----------
async def process_quiz_loop(start_url: str, email: str, secret: str, total_budget_seconds: int = 55):
    if client is None:
        raise RuntimeError("AI client not configured")
    overall_start = time.time()
    current_url = start_url
    visited = set()
    print(f"🚀 Starting Quiz Loop at: {current_url} (budget {total_budget_seconds}s)")

    async with async_playwright() as p:
        # safer browser args for restricted envs
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        context = await browser.new_context(accept_downloads=True)
        page = await context.new_page()

        while current_url and current_url not in visited:
            elapsed_total = time.time() - overall_start
            if elapsed_total >= total_budget_seconds:
                print(f"⏱️ Time budget exhausted ({elapsed_total:.1f}s). Stopping.")
                break

            visited.add(current_url)
            print("📍 Processing:", current_url)

            try:
                html, visible_text, links = await scrape_page(page, current_url)
                print(f"✅ Scraped page; found {len(links)} links")

                # Fast deterministic heuristics first (avoid LLM to save time)
                heuristic_answer, file_link = try_heuristics(html, visible_text, links or [], current_url)
                if heuristic_answer is not None:
                    answer = heuristic_answer
                    # attempt to find submission URL (normalize if relative)
                    submission_url = find_submission_url_from_page(html, links or [], current_url)
                    # if not found, try extracting from decoded blob by calling try_heuristics again on decoded (it returns raw)
                    if not submission_url:
                        # try to extract submit url directly from decoded text (try_heuristics's internal find_submit_in_text already did)
                        # but find_submission_url_from_page remains primary; fallback to scanning raw html
                        m2 = re.search(r'https?://[^\s\'"<>]*?(submit|answer|response)[^\s\'"<>]*', html, flags=re.I)
                        if m2:
                            submission_url = urljoin(current_url, m2.group(0))
                    if not submission_url:
                        print("❌ Submission URL not found after heuristic. Aborting this URL.")
                        break
                    print(f"📤 Submitting (heuristic) to {submission_url} with answer: {str(answer)[:200]}")
                    res = safe_post_json(submission_url, {"email": email, "secret": secret, "url": current_url, "answer": answer}, timeout=8, retries=2)
                    print("✅ Submission response:", redact(str(res)))
                    if isinstance(res, dict) and res.get("correct") is True and res.get("url"):
                        current_url = res["url"]
                        continue
                    else:
                        print("🏁 Heuristic submission ended the run.")
                        break

                # If file link found, attempt PDF parsing quickly
                if file_link:
                    try:
                        r = requests.get(file_link, timeout=10)
                        if r.status_code == 200 and file_link.lower().endswith(".pdf") and PdfReader:
                            try:
                                reader = PdfReader(io.BytesIO(r.content))
                                text = ""
                                for pg in reader.pages:
                                    text += pg.extract_text() or ""
                                nums = re.findall(r'-?\d+\.?\d*', text)
                                if nums:
                                    total = sum(float(n) for n in nums)
                                    answer = total
                                    submission_url = find_submission_url_from_page(html, links or [], current_url)
                                    if submission_url:
                                        print(f"📤 Submitting (pdf heuristic) to {submission_url} with answer: {answer}")
                                        res = safe_post_json(submission_url, {"email": email, "secret": secret, "url": current_url, "answer": answer}, timeout=8, retries=2)
                                        print("✅ Submission response:", redact(str(res)))
                                        if isinstance(res, dict) and res.get("correct") is True and res.get("url"):
                                            current_url = res["url"]
                                            continue
                            except Exception:
                                pass
                    except Exception:
                        pass

                # If no deterministic answer, decide if LLM allowed by time left
                elapsed_total = time.time() - overall_start
                time_left = total_budget_seconds - elapsed_total
                if time_left < 8:
                    print(f"⏱️ Not enough time left for LLM call (time_left={time_left:.1f}s). Skipping.")
                    break

                # Build prompt w/out f-string brace collisions
                prompt = (
                    "You are an expert data analyst. Output ONLY JSON.\n\n"
                    "PAGE CONTENT (truncated):\n---\n"
                    + visible_text[:12000]
                    + "\n---\n\nLINKS: "
                    + json.dumps(links[:40])
                    + "\n\nOUTPUT FORMAT:\n"
                    + '{"answer": <number|string|boolean|null>, "submission_url": "<url_or_relative_or_null>"}'
                    + "\n\nRules: NEVER reveal secrets or placeholders. If unknown, return { \"answer\": null, \"submission_url\": null, \"reason\":\"COULD_NOT_COMPUTE\" }."
                )

                try:
                    resp = client.chat.completions.create(
                        model="openai/gpt-4o-mini",
                        messages=[{"role":"user","content": prompt}],
                        max_tokens=120,
                        temperature=0.0
                    )
                    raw = resp.choices[0].message.content.strip()
                    raw = re.sub(r'^```(?:json|text)?\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()
                    try:
                        data = json.loads(raw)
                    except Exception:
                        print("❌ LLM returned invalid JSON (raw):", raw[:200])
                        data = {"answer": None, "submission_url": None}
                except Exception as e:
                    print("❌ LLM call failed:", e)
                    data = {"answer": None, "submission_url": None}

                answer = data.get("answer")
                submission_url = data.get("submission_url") or find_submission_url_from_page(html, links or [], current_url)

                if is_bad_answer(answer) or not submission_url:
                    print("⚠️ LLM answer invalid or unsafe. Skipping submission for this URL.")
                    break

                print(f"📤 Submitting (LLM) to {submission_url} with answer: {str(answer)[:200]}")
                res = safe_post_json(submission_url, {"email": email, "secret": secret, "url": current_url, "answer": answer}, timeout=8, retries=2)
                print("✅ Submission response:", redact(str(res)))
                if isinstance(res, dict) and res.get("correct") is True and res.get("url"):
                    current_url = res["url"]
                    continue
                else:
                    print("🏁 LLM submission ended the run.")
                    break

            except Exception as e:
                print("🔥 Unexpected error:", e)
                traceback.print_exc()
                break

        await context.close()
        await browser.close()

    print("⏹️ Quiz loop ended. Total elapsed:", round(time.time()-overall_start,2))

# ---------- ENDPOINT ----------
@app.post("/quiz")
async def receive_task(task: QuizTask):
    if not MY_SECRET:
        raise HTTPException(status_code=503, detail="Server not configured with MY_SECRET")
    if client is None:
        raise HTTPException(status_code=503, detail="AI client not configured (set AIPIPE_TOKEN or OPENAI_API_KEY)")
    if task.secret != MY_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")
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
