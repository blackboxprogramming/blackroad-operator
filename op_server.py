import json, time, requests
from typing import List, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
from playwright.sync_api import sync_playwright

LLM_URL = "http://127.0.0.1:11434/api/generate"   # Ollama-compatible (optional)
LLM_MODEL = "qwen2.5"

app = FastAPI(title="Operator")

class Task(BaseModel):
    goal: str
    site: str | None = None

def llm_plan(goal: str) -> List[Dict[str, Any]]:
    prompt = f"""You are an Operator planner.
Goal: {goal}
Return a JSON array (max 6 steps). Each step:
{{"action":"open|type|click|press|wait|read","target":"CSS selector or URL","value":""}}."""
    try:
        r = requests.post(LLM_URL, json={"model": LLM_MODEL, "prompt": prompt, "stream": False}, timeout=30)
        text = r.json().get("response","[]")
        plan = json.loads(text) if text.strip().startswith("[") else []
        if not plan:
            raise ValueError("non-JSON plan")
        return plan[:6]
    except Exception:
        return [
            {"action":"open","target":"https://duckduckgo.com","value":""},
            {"action":"type","target":"input[name=q]","value":goal},
            {"action":"press","target":"","value":"Enter"},
            {"action":"wait","target":"","value":"3s"},
            {"action":"read","target":"","value":""}
        ]

def run_browser_plan(plan: List[Dict[str, Any]]) -> Dict[str, Any]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        log = []
        try:
            for step in plan:
                a = step.get("action",""); t = step.get("target",""); v = step.get("value","")
                if a == "open":
                    page.goto(t or "https://duckduckgo.com", wait_until="domcontentloaded")
                elif a == "type":
                    page.fill(t, v, timeout=10000)
                elif a == "click":
                    page.click(t, timeout=10000)
                elif a == "press":
                    page.keyboard.press(v or "Enter")
                elif a == "wait":
                    time.sleep(int(v.replace("s","")) if isinstance(v,str) and v.endswith("s") else 3)
                elif a == "read":
                    log.append({"read": page.content()[:2000]})
                log.append({"done": step})
            screenshot = page.screenshot(full_page=False)
            return {"ok": True, "steps": log, "screenshot_hex": screenshot.hex()[:256]}
        finally:
            browser.close()

@app.post("/tasks")
def run_task(t: Task):
    plan = llm_plan(t.goal)
    result = run_browser_plan(plan)
    return {"goal": t.goal, "plan": plan, "result": result}

@app.get("/health")
def health():
    return {"ok": True}
