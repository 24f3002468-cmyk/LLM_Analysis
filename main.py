import os
import json
import requests
import uvicorn
import io
import sys
import traceback
import re
import pandas as pd
import pypdf
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from playwright.async_api import async_playwright
from openai import OpenAI

# --- INITIALIZE APP (Ye line miss ho gayi thi) ---
app = FastAPI()

# --- 1. CONFIGURATION ---
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN")
MY_SECRET = os.environ.get("MY_SECRET", "hemant_secret_123")

if not AIPIPE_TOKEN:
    print("⚠️ WARNING: AIPIPE_TOKEN not found! Check Render Environment Variables.")

# AI Pipe Client
client = OpenAI(
    api_key=AIPIPE_TOKEN,
    base_url="https://aipipe.org/openrouter/v1"
)

class QuizTask(BaseModel):
    email: str
    secret: str
    url: str

# --- 2. WELCOME PAGE ---
@app.get("/")
def home():
    return {
        "status": "Active",
        "message": "AI Agent Ready. Send POST to /quiz",
        "capabilities": ["Web Scraping", "PDF Reading", "Turbo Mode"]
    }

# --- 3. HELPER: EXECUTE CODE ---
def execute_python_code(code_str: str):
    old_stdout = sys.stdout
    redirected_output = sys.stdout = io.StringIO()
    exec_globals = {
        '__name__': '__main__',
        'requests': requests, 'json': json, 're': re,
        'pd': pd, 'pypdf': pypdf, 'io': io
    }
    try:
        exec(code_str, exec_globals)
        return redirected_output.getvalue().strip()
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        sys.stdout = old_stdout

# --- 4. MAIN LOGIC (OPTIMIZED TURBO MODE) ---
async def process_quiz_loop(start_url: str, email: str, secret: str):
    current_url = start_url
    visited_urls = set()
    print(f"🚀 Starting Quiz Loop at: {current_url}")

    while current_url and current_url not in visited_urls:
        visited_urls.add(current_url)
        print(f"\n📍 Processing Level: {current_url}")
        try:
            # A. SCRAPE (TURBO BROWSER LAUNCH)
            async with async_playwright() as p:
                print("⏳ Launching Browser (Turbo Mode)...")
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-setuid-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-accelerated-2d-canvas",
                        "--no-first-run",
                        "--no-zygote",
                        "--single-process",
                        "--disable-gpu"
                    ]
                )
                page = await browser.new_page()
                
                print(f"🌐 Navigating to {current_url}...")
                # 30 sec timeout taaki latke nahi
                await page.goto(current_url, timeout=30000, wait_until="domcontentloaded")
                
                try: await page.wait_for_selector("body", timeout=5000)
                except: pass
                
                visible_text = await page.inner_text("body")
                links = await page.evaluate("""() => {
                    return Array.from(document.querySelectorAll('a')).map(a => a.href);
                }""")
                await browser.close()
                print(f"✅ Page Scraped. Found {len(links)} links.")

            # B. ASK AI (STRICT MODE)
            prompt = f"""
            You are an expert Python Data Analyst.
            PAGE CONTENT:
            ---
            {visible_text[:6000]}
            ---
            AVAILABLE LINKS: {links}
            
            YOUR TASK:
            1. Identify the Question.
            2. Write Python code to CALCULATE the answer.
            3. Identify the Submission URL.
            
            CRITICAL RULES:
            - **DO NOT SUBMIT DATA**: Do NOT use `requests.post`. Only calculate.
            - **NO PLACEHOLDERS**: Use REAL links from the list.
            - **OUTPUT FORMAT**: JSON string: {{"answer": <value>, "submission_url": "<url>"}}
            - **ANSWER TYPE**: String or Number only.
            - Output ONLY raw Python code.
            """

            print(f"🤖 Asking AI Pipe...")
            response = client.chat.completions.create(
                model="openai/gpt-4o-mini", 
                messages=[{"role": "user", "content": prompt}]
            )
            ai_code = response.choices[0].message.content.replace("```python", "").replace("```", "").strip()
            print(f"📝 AI Code Generated.")

            # C. EXECUTE
            execution_result = execute_python_code(ai_code)
            print(f"⚡ Result: {execution_result}")

            # D. PARSE & SUBMIT
            submit_url = None
            answer = None
            try:
                match = re.search(r'\{.*\}', execution_result, re.DOTALL)
                if match:
                    data = json.loads(match.group())
                    answer = data.get("answer")
                    submit_url = data.get("submission_url")
                    if submit_url and not submit_url.startswith("http"):
                        from urllib.parse import urljoin
                        submit_url = urljoin(current_url, submit_url)
            except: pass

            if not submit_url:
                print("❌ Submission URL not found."); break
            
            if isinstance(answer, dict): answer = str(answer)

            payload = {"email": email, "secret": secret, "url": current_url, "answer": answer}
            print(f"📤 Submitting to {submit_url} with answer: {answer}")
            
            res = requests.post(submit_url, json=payload, timeout=10).json()
            print(f"✅ Response: {res}")
            
            if res.get("correct") == True and "url" in res:
                current_url = res["url"]
            else:
                print("🏁 Quiz Finished.")
                current_url = None

        except Exception as e:
            print(f"🔥 Error: {e}"); traceback.print_exc(); current_url = None

# --- 5. API ENDPOINT ---
@app.post("/quiz")
async def receive_task(task: QuizTask, background_tasks: BackgroundTasks):
    if task.secret != MY_SECRET: raise HTTPException(status_code=403, detail="Invalid Secret")
    background_tasks.add_task(process_quiz_loop, task.url, task.email, task.secret)
    return {"message": "AI Agent started."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
