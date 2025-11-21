# main.py
import os
import re
import io
import json
import time
import traceback
import tempfile
import requests
import asyncio
from typing import List, Optional, Tuple, Any
from urllib.parse import urljoin, urlparse

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Playwright (installed in Dockerfile)
from playwright.async_api import async_playwright, Page

# Optional helpers
from bs4 import BeautifulSoup
import pandas as pd
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

# ---------- CONFIG ----------
MY_SECRET = os.environ.get("MY_SECRET")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DEFAULT_BUDGET_SECONDS = int(os.environ.get("DEFAULT_BUDGET_SECONDS", "110"))
MAX_CONCURRENT_BROWSERS = int(os.environ.get("MAX_CONCURRENT_BROWSERS", "1"))

if not MY_SECRET:
    print("⚠️ WARNING: MY_SECRET not set. Use a dev secret for testing.")

app = FastAPI(title="LLM Analysis Quiz Agent")

# Concurrency semaphore
from asyncio import Semaphore
BROWSER_SEMAPHORE = Semaphore(MAX_CONCURRENT_BROWSERS)

# ---------- Models ----------
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
    # Synchronous helper (used via asyncio.to_thread)
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
        except Exception as e:
            last_exc = e
            if i == retries - 1:
                raise
            time.sleep(0.5 * (2 ** i))
    raise last_exc

def find_submission_url_from_page(html: str, links: List[str], base_url: str) -> Optional[str]:
    # 1) links with submit-like keywords
    for l in links or []:
        if not l:
            continue
        low = l.lower()
        if any(k in low for k in ("submit", "answer", "response", "/submit")):
            return urljoin(base_url, l)
    # 2) form action
    m = re.search(r'<form[^>]+action=["\']([^"\']+)["\']', html, flags=re.I)
    if m:
        return urljoin(base_url, m.group(1))
    # 3) explicit urls in scripts
    m2 = re.search(r'https?://[^\s\'"<>]+/(?:submit|answer|response)[^\s\'"<>]*', html, flags=re.I)
    if m2:
        return m2.group(0)
    # 4) fallback absolute link
    for l in links or []:
        if l and l.startswith("http"):
            return l
    return None

# ---------- Heuristics ----------
def likely_has_quiz(html: str, visible_text: str, links: List[str]) -> bool:
    text = (visible_text or "") + " " + (html or "")
    text_low = text.lower()
    if any(k in text_low for k in ("value", "sum", "total", "answer", "question", "submit", "download", "file")):
        return True
    if "<form" in (html or "").lower():
        return True
    for l in links or []:
        if l and any(l.lower().endswith(ext) for ext in (".pdf", ".csv", ".xlsx", ".xls", ".mp3", ".wav")):
            return True
    return False

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

    # JSON blob in page (common when using atob)
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

    # HTML tables via pandas
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
            for col in df.columns:
                colnums = pd.to_numeric(df[col], errors='coerce')
                if colnums.notna().any():
                    submit_url = find_submit_in_text(html)
                    return float(colnums.sum(skipna=True)), (urljoin(base_url, submit_url) if submit_url else None)
    except Exception:
        pass

    # visible text numbers
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

    # downloadable files: pdf/csv/xlsx or audio
    try:
        for l in links or []:
            if l and any(l.lower().endswith(ext) for ext in (".pdf", ".csv", ".xlsx", ".xls", ".mp3", ".wav")):
                return None, l
    except Exception:
        pass

    final_try = find_submit_in_text(html) or find_submit_in_text(visible_text)
    if final_try:
        return None, urljoin(base_url, final_try)

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

def looks_like_secret(s: str) -> bool:
    if not isinstance(s, str):
        return False
    st = s.strip()
    if st == "":
        return False
    if len(st) <= 64 and re.fullmatch(r'[A-Za-z0-9_\-]{3,64}', st):
        return True
    if re.search(r'\b(secret|code|token|key|pass|pwd)\b', st, flags=re.I):
        return True
    if len(st) <= 4:
        return True
    return False

# ---------- OpenAI audio transcription (REST) ----------
def transcribe_audio_bytes_via_openai(audio_bytes: bytes, filename_hint="audio.mp3") -> Optional[str]:
    if not OPENAI_API_KEY:
        print("⚠️ OPENAI_API_KEY not set; cannot transcribe audio.")
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename_hint)[1] or ".mp3", delete=False) as tf:
            tf.write(audio_bytes)
            tmp_path = tf.name
        # Use OpenAI transcription REST endpoint - compatible fallback
        url = "https://api.openai.com/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        files = {"file": open(tmp_path, "rb")}
        data = {"model": "whisper-1"}
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=60)
        files["file"].close()
        if resp.status_code != 200:
            print("⚠️ Transcription API failed:", resp.status_code, resp.text[:200])
            return None
        j = resp.json()
        text = j.get("text") or j.get("transcript") or None
        return text
    except Exception as e:
        print("⚠️ Transcription exception:", e)
        return None

def extract_answer_from_transcript(transcript_text: str):
    if not transcript_text:
        return None
    nums = re.findall(r'-?\d+\.?\d*', transcript_text)
    if nums:
        vals = [float(n) for n in nums]
        return sum(vals)
    txt = transcript_text.strip()
    return txt if len(txt) <= 1000 else txt[:1000]

# ---------- LLM wrapper (light) ----------
def call_llm_and_parse(prompt_text: str):
    # If user has OPENAI_API_KEY, call simple completion to parse JSON
    if not OPENAI_API_KEY:
        raise RuntimeError("LLM client not configured (OPENAI_API_KEY missing)")
    # Minimal safety: system instruction for JSON-only
    system_msg = (
        "You are an expert data analyst. OUTPUT ONLY valid JSON matching: "
        '{"answer": <value|null|number|boolean|string>, "submission_url": "<url_or_null>"} . '
        "Do NOT reveal secrets. If no quiz present return {\"answer\": null, \"submission_url\": null, \"reason\":\"NO_QUIZ\"}."
    )
    try:
        # Use OpenAI chat completions (REST)
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        body = {
            "model": "gpt-4o-mini" if True else "gpt-4o-mini",  # change if needed
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt_text}
            ],
            "temperature": 0.0,
            "max_tokens": 600
        }
        r = requests.post(url, headers=headers, json=body, timeout=30)
        r.raise_for_status()
        j = r.json()
        raw = ""
        try:
            raw = j["choices"][0]["message"]["content"].strip()
        except Exception:
            raw = str(j)[:2000]
    except Exception as e:
        raise RuntimeError(f"LLM invocation error: {e}")

    raw_redacted = redact(raw[:2000])
    print("🧾 LLM raw (redacted):", raw_redacted)

    raw = re.sub(r'^```(?:json|text)?\s*|\s*```$', '', raw, flags=re.MULTILINE).strip()
    jm = re.search(r'(\{.*\})', raw, flags=re.S)
    if not jm:
        raise ValueError("LLM did not return JSON object")
    try:
        data = json.loads(jm.group(1))
    except Exception as e:
        raise ValueError(f"LLM JSON parse error: {e}")

    answer = data.get("answer")
    submission_url = data.get("submission_url")
    return answer, submission_url

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

    links = []
    try:
        links = await page.evaluate("() => Array.from(document.querySelectorAll('a')).map(a => a.href)")
    except Exception:
        links = []

    # extract atob(...) in scripts client-side
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
                        try { out.push(atob(m[1])); } catch(e) {}
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

# ---------- Main quiz loop ----------
async def process_quiz_loop(start_url: str, email: str, secret: str, total_budget_seconds: int = DEFAULT_BUDGET_SECONDS):
    overall_start = time.time()
    current_url = start_url
    visited = set()
    print(f"🚀 Starting Quiz Loop: {current_url} (budget {total_budget_seconds}s)")

    # Concurrency guard
    async with BROWSER_SEMAPHORE:
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

                print("Visiting host:", urlparse(current_url).hostname)
                print("📍 Processing:", current_url)
                try:
                    html, visible_text, links = await scrape_page(page, current_url)
                    print(f"✅ Scraped page; found {len(links)} links")

                    if not likely_has_quiz(html, visible_text, links):
                        print("ℹ️ Page appears to have no quiz content. Skipping.")
                        break

                    # Heuristics-first
                    heuristic_answer, file_or_submit = try_heuristics(html, visible_text, links or [], current_url)
                    if heuristic_answer is not None:
                        submission_url = find_submission_url_from_page(html, links or [], current_url)
                        if not submission_url:
                            print("❌ Submission URL not found after heuristic. Aborting this URL.")
                        else:
                            if is_bad_answer(heuristic_answer) or looks_like_secret(str(heuristic_answer)):
                                print("⚠️ Heuristic answer rejected (invalid or secret-like). Falling back.")
                            else:
                                payload = {"email": email, "secret": secret, "url": current_url, "answer": heuristic_answer}
                                # payload size check
                                pj = json.dumps(payload)
                                if len(pj.encode()) > 1_000_000:
                                    print("⚠️ Payload too large; skipping heuristic submit")
                                else:
                                    try:
                                        print("📤 Heuristic submit:", redact(str(heuristic_answer)))
                                        res = await asyncio.to_thread(safe_post_json, submission_url, payload, 8, 2)
                                        print("Res:", redact(str(res)))
                                        if isinstance(res, dict) and res.get("correct") is True and res.get("url"):
                                            current_url = res["url"]; continue
                                        elif isinstance(res, dict) and res.get("url"):
                                            print("Incorrect but provided next URL; continuing.")
                                            current_url = res["url"]; continue
                                        else:
                                            print("🏁 Heuristic submission ended the run.")
                                            break
                                    except Exception as e:
                                        print("⚠️ Heuristic submission failed:", e)

                    # File or audio handling
                    if file_or_submit:
                        file_link = file_or_submit
                        # AUDIO
                        if file_link.lower().endswith((".mp3", ".wav")):
                            try:
                                r = requests.get(file_link, timeout=20)
                                if r.status_code == 200:
                                    transcript = transcribe_audio_bytes_via_openai(r.content, filename_hint=file_link)
                                    print("🔊 Transcript (redacted):", redact((transcript or "")[:500]))
                                    answer_candidate = extract_answer_from_transcript(transcript) if transcript else None
                                    if answer_candidate and not (is_bad_answer(answer_candidate) or looks_like_secret(str(answer_candidate))):
                                        submission_url = find_submission_url_from_page(html, links or [], current_url)
                                        if submission_url:
                                            payload = {"email": email, "secret": secret, "url": current_url, "answer": answer_candidate}
                                            pj = json.dumps(payload)
                                            if len(pj.encode()) > 1_000_000:
                                                print("⚠️ Payload too large; skipping audio submit")
                                            else:
                                                res = await asyncio.to_thread(safe_post_json, submission_url, payload, 12, 2)
                                                print("✅ Submission response:", redact(str(res)))
                                                if isinstance(res, dict) and res.get("url"):
                                                    current_url = res.get("url"); continue
                                                else:
                                                    print("🏁 Audio submission ended the run.")
                                                    break
                            except Exception as e:
                                print("⚠️ Audio handling failed:", e)

                        # PDF handling
                        if file_link.lower().endswith(".pdf") and PdfReader:
                            try:
                                r = requests.get(file_link, timeout=15)
                                if r.status_code == 200:
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
                                        if submission_url and not is_bad_answer(total):
                                            payload = {"email": email, "secret": secret, "url": current_url, "answer": total}
                                            pj = json.dumps(payload)
                                            if len(pj.encode()) > 1_000_000:
                                                print("⚠️ Payload too large; skipping pdf submit")
                                            else:
                                                res = await asyncio.to_thread(safe_post_json, submission_url, payload, 8, 2)
                                                print("✅ PDF submission response:", redact(str(res)))
                                                if isinstance(res, dict) and res.get("url"):
                                                    current_url = res.get("url"); continue
                                                else:
                                                    print("🏁 PDF submission ended the run.")
                                                    break
                            except Exception as e:
                                print("⚠️ PDF handling failed:", e)

                    # LLM fallback
                    elapsed_total = time.time() - overall_start
                    time_left = total_budget_seconds - elapsed_total
                    if time_left < 10:
                        print("⏱️ Not enough time left for LLM. Stopping.")
                        break

                    # Build prompt
                    prompt = (
                        "You are an expert Python data analyst. Output ONLY valid JSON (no markdown):\n\n"
                        "PAGE VISIBLE TEXT (truncated):\n---\n" + (visible_text[:12000] or "") + "\n---\n\n"
                        "LINKS: " + json.dumps(links[:40]) + "\n\n"
                        'OUTPUT FORMAT: {"answer": <number|string|boolean|null>, "submission_url": "<url_or_relative_or_null>"}\n\n'
                        'If cannot compute confidently, return {"answer": null, "submission_url": null, "reason":"COULD_NOT_COMPUTE"}.'
                    )

                    try:
                        answer, submission_url = await asyncio.get_event_loop().run_in_executor(None, call_llm_and_parse, prompt)
                    except Exception as e:
                        print("❌ LLM call failed or returned invalid/suspicious output:", e)
                        break

                    if isinstance(submission_url, str) and submission_url:
                        submission_url = urljoin(current_url, submission_url)

                    if is_bad_answer(answer) or not submission_url:
                        print("⚠️ LLM answer invalid/unsafe or missing submission_url. Skipping submission.")
                        break

                    payload = {"email": email, "secret": secret, "url": current_url, "answer": answer}
                    pj = json.dumps(payload)
                    if len(pj.encode()) > 1_000_000:
                        print("⚠️ Payload too large; aborting LLM submit")
                        break

                    print(f"📤 Submitting (LLM) to {submission_url} with answer: {redact(str(answer)[:200])}")
                    try:
                        res = await asyncio.to_thread(safe_post_json, submission_url, payload, 8, 2)
                        print("✅ Submission response:", redact(str(res)))
                        if isinstance(res, dict) and res.get("correct") is True and res.get("url"):
                            current_url = res["url"]; continue
                        elif isinstance(res, dict) and res.get("url"):
                            print("Incorrect but chaining URL provided; continuing.")
                            current_url = res["url"]; continue
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
