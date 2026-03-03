import os, json, time, requests
from typing import List, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel, Field
from playwright.sync_api import sync_playwright
import psycopg2, psycopg2.extras
from app_bootstrap import PI_NODES, pi_preflight

# --- config from environment ---
OP_DB_DSN = os.getenv("OP_DB_DSN", "postgresql:///operator")

# Pi cluster LLM — Ollama runs on the Pi nodes, not an external provider
PI_LLM_HOST = os.getenv("OP_PI_LLM_HOST", "192.168.4.49")  # alice by default
PI_LLM_PORT = os.getenv("OP_PI_LLM_PORT", "11434")
PI_LLM_URL  = os.getenv("OP_LLM_URL", f"http://{PI_LLM_HOST}:{PI_LLM_PORT}/api/generate")
LLM_MODEL   = os.getenv("OP_LLM_MODEL", "qwen2.5:0.5b")

app = FastAPI(title="BlackRoad Operator — Pi Cluster")

class Task(BaseModel):
    goal: str
    site: str | None = None

# --- DB logging ---
def log_event(role: str, event: dict) -> None:
    try:
        conn = psycopg2.connect(OP_DB_DSN)
        with conn, conn.cursor() as cur:
            psycopg2.extras.register_default_jsonb(cur)
            cur.execute(
                "INSERT INTO op_events(role, event) VALUES (%s, %s::jsonb)",
                (role, json.dumps(event))
            )
    except Exception:
        pass

# --- Pi cluster LLM readiness probe ---
def wait_for_llm(timeout=20) -> bool:
    tags_url = PI_LLM_URL.replace('/api/generate', '/api/tags')
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(tags_url, timeout=2)
            if r.ok:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False

# --- planner (runs on Pi cluster LLM) ---
def llm_plan(goal: str) -> List[Dict[str, Any]]:
    prompt = f"""You are an Operator planner.
Goal: {goal}
Return a JSON array (max 6 steps). Each step:
{{"action":"open|type|click|press|wait|read","target":"CSS selector or URL","value":""}}."""
    try:
        r = requests.post(PI_LLM_URL, json={"model": LLM_MODEL, "prompt": prompt, "stream": False}, timeout=30)
        r.raise_for_status()
        text = r.json().get("response", "[]")
        plan = json.loads(text) if text.strip().startswith("[") else []
        if not plan:
            raise ValueError("empty or non-JSON plan")
        return plan[:6]
    except Exception as e:
        log_event("planner_error", {"error": str(e)})
        return [
            {"action": "open", "target": "https://duckduckgo.com", "value": ""},
            {"action": "type", "target": "input[name=q]", "value": goal},
            {"action": "press", "target": "", "value": "Enter"},
            {"action": "wait", "target": "", "value": "3s"},
            {"action": "read", "target": "", "value": ""}
        ]

# --- executor ---
def run_browser_plan(plan: List[Dict[str, Any]]) -> Dict[str, Any]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        log = []
        try:
            for step in plan:
                a = step.get("action", ""); t = step.get("target", ""); v = step.get("value", "")
                if a == "open":
                    page.goto(t or "https://duckduckgo.com", wait_until="domcontentloaded")
                elif a == "type":
                    page.fill(t, v, timeout=10000)
                elif a == "click":
                    page.click(t, timeout=10000)
                elif a == "press":
                    page.keyboard.press(v or "Enter")
                elif a == "wait":
                    try:
                        secs = int(v.replace("s", "")) if isinstance(v, str) and v.endswith("s") else 3
                    except Exception:
                        secs = 3
                    time.sleep(secs)
                elif a == "read":
                    log.append({"read": page.content()[:2000]})
                log.append({"done": step})
            screenshot = page.screenshot(full_page=False)
            return {"ok": True, "steps": log, "screenshot_hex": screenshot.hex()[:256]}
        finally:
            browser.close()

# --- routes ---
@app.get("/health")
def health():
    return {"ok": True, "cluster": "pi"}

@app.get("/llm/health")
def llm_health():
    try:
        tags_url = PI_LLM_URL.replace('/api/generate', '/api/tags')
        r = requests.get(tags_url, timeout=2)
        return {"ok": r.ok, "host": PI_LLM_HOST}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/cluster/health")
def cluster_health():
    from tls_guard import check_pi_node
    status = {}
    for name, ip in PI_NODES.items():
        status[name] = {"ip": ip, "reachable": check_pi_node(ip)}
    return {"ok": any(n["reachable"] for n in status.values()), "nodes": status}

@app.post("/tasks")
def run_task(t: Task):
    if not wait_for_llm(10):
        log_event("planner_error", {"error": "Pi LLM not ready", "url": PI_LLM_URL})
    log_event("user", {"goal": t.goal, "site": t.site})
    plan = llm_plan(t.goal)
    if t.site:
        plan = [{"action": "open", "target": t.site, "value": ""}] + plan
    log_event("planner", {"plan": plan})
    result = run_browser_plan(plan)
    log_event("executor", {"result": {"ok": result.get("ok", False), "n_steps": len(result.get("steps", []))}})
    return {"goal": t.goal, "plan": plan, "result": result}

# --- Simulation tool: Gymnasium (MuJoCo / others) ---
from PIL import Image
import numpy as np, io, base64, gymnasium as gym

class SimReq(BaseModel):
    env_id: str = Field(..., description="Gymnasium environment ID, e.g. 'HalfCheetah-v5'")
    steps: int = Field(50, ge=1, le=2000)
    render: bool = Field(True, description="Return a preview frame (PNG base64)")
    seed: int | None = Field(None)
    mujoco_gl: str | None = Field(None, description="Override MUJOCO_GL: 'egl' or 'osmesa'")

def _encode_png(frame: np.ndarray) -> str:
    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")

@app.post("/tools/sim.run")
def sim_run(req: SimReq):
    if req.render and req.mujoco_gl and not os.getenv("MUJOCO_GL"):
        os.environ["MUJOCO_GL"] = req.mujoco_gl

    render_mode = "rgb_array" if req.render else None
    try:
        env = gym.make(req.env_id, render_mode=render_mode)
    except Exception as e:
        log_event("sim_error", {"env_id": req.env_id, "error": str(e)})
        return {"ok": False, "error": f"failed to make env: {e}"}

    preview_b64 = None
    total_r = 0.0
    steps = 0
    try:
        obs, info = env.reset(seed=req.seed)
        for _ in range(req.steps):
            action = env.action_space.sample()
            obs, r, terminated, truncated, info = env.step(action)
            total_r += float(r)
            steps += 1
            if req.render:
                frame = env.render()
                if frame is not None and preview_b64 is None:
                    preview_b64 = _encode_png(frame)
            if terminated or truncated:
                break
        ok = True
    except Exception as e:
        ok = False
        log_event("sim_error", {"env_id": req.env_id, "error": str(e)})
        return {"ok": False, "error": str(e)}
    finally:
        try:
            env.close()
        except Exception:
            pass

    payload = {
        "ok": ok,
        "env_id": req.env_id,
        "steps": steps,
        "return_sum": total_r,
    }
    if preview_b64:
        payload["preview_png_base64"] = preview_b64

    log_event("sim", {"env_id": req.env_id, "steps": steps, "return_sum": total_r, "render": req.render})
    return payload

# --- Code runner tool (Docker / Python) ---
from pathlib import Path
import tempfile, shutil, subprocess

class CodeRunReq(BaseModel):
    language: str = Field("python", description="Only 'python' supported for now")
    code: str
    deps: list[str] | None = None
    timeout: int = Field(20, ge=1, le=300)
    return_files: list[str] | None = None

@app.post("/tools/code.run")
def code_run(req: CodeRunReq):
    if req.language.lower() != "python":
        return {"ok": False, "error": "only python is supported for now"}

    tmp = Path(tempfile.mkdtemp(prefix="op-code-"))
    try:
        (tmp / "main.py").write_text(req.code)
        if req.deps:
            (tmp / "requirements.txt").write_text("\n".join(req.deps) + "\n")

        cmd = [
            "docker", "run", "--rm",
            "--network", "none",
            "--pids-limit", "256",
            "--memory", "512m",
            "-v", f"{tmp}:/work",
            "-w", "/work",
            "python:3.11-slim", "bash", "-lc",
            "set -e; "
            "python -m venv .venv; . .venv/bin/activate; "
            "if [ -f requirements.txt ]; then pip -q install --no-cache-dir -r requirements.txt; fi; "
            "python main.py"
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=req.timeout)
        res = {"ok": p.returncode == 0, "exit_code": p.returncode,
               "stdout": p.stdout[-8000:], "stderr": p.stderr[-8000:]}

        if req.return_files:
            out = {}
            for name in req.return_files:
                fp = tmp / name
                if fp.exists() and fp.is_file():
                    out[name] = base64.b64encode(fp.read_bytes()).decode("ascii")
            if out:
                res["files_base64"] = out

        log_event("code_run", {"ok": res["ok"], "exit_code": res["exit_code"]})
        return res

    except subprocess.TimeoutExpired:
        log_event("code_run", {"ok": False, "error": "timeout"})
        return {"ok": False, "error": "timeout"}
    except Exception as e:
        log_event("code_run", {"ok": False, "error": str(e)})
        return {"ok": False, "error": str(e)}
    finally:
        try:
            shutil.rmtree(tmp)
        except Exception:
            pass

# --- Git PR tool ---
class FileSpec(BaseModel):
    path: str
    content: str

class GitPrReq(BaseModel):
    branch: str
    title: str
    body: str = ""
    commit_message: str = "Operator update"
    files: list[FileSpec]
    repo_path: str | None = None

def _run(cwd, *cmd):
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr

@app.post("/tools/git.pr")
def git_pr(req: GitPrReq):
    repo = Path(req.repo_path or "/home/pi/operator")
    if not (repo / ".git").exists():
        return {"ok": False, "error": f"not a git repo: {repo}"}
    try:
        _run(repo, "git", "fetch", "origin")
        _run(repo, "git", "checkout", "-B", req.branch)

        for f in req.files:
            fp = (repo / f.path)
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(f.content)

        _run(repo, "git", "add", "-A")
        rc, out, err = _run(repo, "git", "commit", "-m", req.commit_message)

        rc, out, err = _run(repo, "git", "push", "-u", "origin", req.branch)
        if rc != 0:
            return {"ok": False, "error": err or out}

        rc, out, err = _run(repo, "gh", "pr", "create", "--title", req.title, "--body", req.body, "--head", req.branch)
        pr_url = out.strip() if rc == 0 else ""
        log_event("git_pr", {"branch": req.branch, "pr_url": pr_url, "rc": rc})
        return {"ok": rc == 0, "pr_url": pr_url, "gh_out": out, "gh_err": err}
    except Exception as e:
        log_event("git_pr", {"ok": False, "error": str(e)})
        return {"ok": False, "error": str(e)}
