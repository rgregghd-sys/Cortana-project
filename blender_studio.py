#!/usr/bin/env python3
"""
Blender AI Studio
-----------------
Natural language -> Blender Python script -> headless render -> Gemini Vision review -> iterative refinement.

Run:  cd ~/Cortana_Project && venv/bin/python3 blender_studio.py
Open: http://localhost:8082
"""

import asyncio, base64, json, os, subprocess, sys, tempfile, textwrap, time, warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Optional, List

# ── deps ──────────────────────────────────────────────────────────────────────
try:
    import google.generativeai as genai
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    from dotenv import load_dotenv
except ImportError as e:
    sys.exit(f"Missing dependency: {e}\nRun: pip install google-generativeai fastapi uvicorn python-dotenv")

# ── config ────────────────────────────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")
genai.configure(api_key=os.getenv("GEMINI_API_KEY", ""))

BLENDER_BIN  = os.getenv("BLENDER_BIN", "blender")
WORKSPACE    = Path(__file__).parent / "agent_workspace" / "blender_studio"
WORKSPACE.mkdir(parents=True, exist_ok=True)
RENDER_DIR   = WORKSPACE / "renders"
RENDER_DIR.mkdir(exist_ok=True)
SCRIPTS_DIR  = WORKSPACE / "scripts"
SCRIPTS_DIR.mkdir(exist_ok=True)

MAX_REFINE   = 0          # visual refinement loops after first render (0 = fastest)
RENDER_RES   = 256        # pixels (square) — bump to 512 for final quality
RENDER_TIMEOUT = 60       # seconds

# ── system prompts ────────────────────────────────────────────────────────────
SCRIPT_SYSTEM = textwrap.dedent("""\
    You are an expert Blender 3.4 Python (bpy) script writer for headless rendering.

    STRICT RULES:
    1. Begin with: import bpy
    2. First thing: clear default scene
       bpy.ops.object.select_all(action='SELECT')
       bpy.ops.object.delete(use_global=False)
    3. Always add a camera AND at least one light.
    4. OUTPUT_PATH is injected as a variable before your script runs — use it as the render filepath.
    5. End with: bpy.ops.render.render(write_still=True)
    6. Use EEVEE (fast): bpy.context.scene.render.engine = 'BLENDER_EEVEE'
    7. Resolution: bpy.context.scene.render.resolution_x = 512 / resolution_y = 512
    8. Format: bpy.context.scene.render.image_settings.file_format = 'PNG'
    9. Low sample count for speed:
       bpy.context.scene.eevee.taa_render_samples = 16
       bpy.context.scene.eevee.use_soft_shadows = False
       bpy.context.scene.eevee.use_ssr = False
   10. Return ONLY the Python code — no markdown fences, no explanations.

    STANDARD CAMERA SETUP:
        bpy.ops.object.camera_add(location=(7, -7, 5))
        cam = bpy.context.object
        cam.rotation_euler = (1.1, 0, 0.785)
        bpy.context.scene.camera = cam

    STANDARD LIGHTING:
        bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
        sun = bpy.context.object
        sun.data.energy = 5
""")

VISION_SYSTEM = textwrap.dedent("""\
    You are a visual QA reviewer for Blender 3D renders.
    Compare the rendered image against the original user request.
    Respond ONLY with valid JSON (no markdown):
    {"matches": true, "score": 8, "issues": [], "suggestions": []}
    - matches: true if render reasonably matches the request
    - score: 0-10 quality score
    - issues: list of specific problems
    - suggestions: concrete improvements (Blender-actionable)
""")

REFINE_SYSTEM = textwrap.dedent("""\
    You are an expert Blender 3.4 Python scripter.
    You receive: the user's request, the previous script, and visual QA feedback.
    Generate a corrected and improved script that addresses all identified issues.
    Return ONLY the Python code — no markdown, no explanation.
""")

EDIT_SYSTEM = textwrap.dedent("""\
    You are an expert Blender 3.4 Python scripter.
    You receive: a new edit instruction, the current working script, and optionally the previous render.
    Modify the script to apply the requested changes while preserving everything else.
    OUTPUT_PATH is pre-injected — keep using it.
    Return ONLY the updated Python code — no markdown, no explanation.
""")

# ── Provider router (Groq -> OpenRouter -> Gemini fallback) ───────────────────
from cortana.providers.router import ProviderRouter as _ProviderRouter
_router = _ProviderRouter()          # uses config.PROVIDER_ORDER + API keys from .env
_GEMINI_MODEL = "gemini-2.0-flash"  # used only for vision calls

def _ask(system: str, user: str, image_path: Optional[Path] = None) -> str:
    """
    Call the AI with an optional render image.
    - Vision calls (image present): routed through think_vision (Gemini only).
    - Text calls: routed through think_simple (Groq -> OpenRouter -> Gemini).
    """
    if image_path and image_path.exists():
        # base64-encode for the vision API
        img_b64 = base64.b64encode(image_path.read_bytes()).decode()
        return _router.think_vision(img_b64, user, system=system, max_tokens=1024)
    else:
        return _router.think_simple(user, system=system, max_tokens=4096)

# ── Blender runner ────────────────────────────────────────────────────────────
def _run_blender(script_text: str, render_path: Path) -> tuple[bool, str]:
    """
    Inject OUTPUT_PATH, write script to disk, run Blender headless.
    Returns (success, log_text).
    """
    injected = f'OUTPUT_PATH = r"{render_path}"\n\n' + script_text
    script_file = SCRIPTS_DIR / f"script_{int(time.time()*1000)}.py"
    script_file.write_text(injected)

    cmd = [BLENDER_BIN, "--background", "--python", str(script_file)]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=RENDER_TIMEOUT,
        )
        log = (result.stdout + "\n" + result.stderr).strip()
        success = render_path.exists() and render_path.stat().st_size > 0
        return success, log
    except subprocess.TimeoutExpired:
        return False, "Blender timed out after 120 seconds."
    except Exception as e:
        return False, f"Blender launch error: {e}"

def _image_b64(path: Path) -> str:
    """Return base64-encoded PNG for embedding in <img src='...'/>."""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()

# ── Pipeline ──────────────────────────────────────────────────────────────────
async def run_generation(prompt: str, ws: WebSocket,
                         edit_script: Optional[str] = None,
                         edit_render: Optional[Path] = None):
    """
    Full generation pipeline:
      1. Generate Blender script from prompt (or edit existing)
      2. Run Blender headless
      3. Vision QA loop (up to MAX_REFINE times)
      4. Stream all events to the WebSocket client
    """

    async def send(event: str, **data):
        await ws.send_text(json.dumps({"event": event, **data}))

    # ── Step 1: generate script ───────────────────────────────────────────────
    await send("status", msg="Generating Blender script...")

    if edit_script:
        # Edit mode: pass existing script + instruction
        user_msg = (
            f"Edit instruction: {prompt}\n\n"
            f"Current script:\n```python\n{edit_script}\n```"
        )
        script = await asyncio.to_thread(_ask, EDIT_SYSTEM, user_msg, edit_render)
    else:
        script = await asyncio.to_thread(_ask, SCRIPT_SYSTEM, f"Create a 3D scene: {prompt}")

    # strip any accidental markdown fences
    if "```" in script:
        lines = script.split("\n")
        script = "\n".join(l for l in lines if not l.strip().startswith("```"))

    await send("script", code=script)

    render_path = RENDER_DIR / f"render_{int(time.time()*1000)}.png"
    current_script = script

    for attempt in range(1 + MAX_REFINE):
        label = f"Attempt {attempt + 1}" if attempt > 0 else "Rendering"
        await send("status", msg=f"{label}: running Blender...")

        success, log = await asyncio.to_thread(_run_blender, current_script, render_path)
        await send("log", text=log[-3000:] if len(log) > 3000 else log)

        if not success:
            if attempt < MAX_REFINE:
                await send("status", msg=f"Render failed — asking AI to fix...")
                fix_prompt = (
                    f"The script failed to produce a render.\n"
                    f"Blender log:\n{log[-1000:]}\n\n"
                    f"Original request: {prompt}\n\n"
                    f"Previous script:\n{current_script}"
                )
                current_script = await asyncio.to_thread(_ask, REFINE_SYSTEM, fix_prompt)
                if "```" in current_script:
                    lines = current_script.split("\n")
                    current_script = "\n".join(l for l in lines if not l.strip().startswith("```"))
                await send("script", code=current_script)
                render_path = RENDER_DIR / f"render_{int(time.time()*1000)}.png"
                continue
            else:
                await send("error", msg="Render failed after all attempts. Check the log.")
                return

        # ── render succeeded ──────────────────────────────────────────────────
        img_b64 = _image_b64(render_path)
        await send("render", image=img_b64, path=str(render_path))

        # ── Vision QA ─────────────────────────────────────────────────────────
        await send("status", msg="AI reviewing render...")
        qa_prompt = (
            f"Original request: \"{prompt}\"\n"
            f"Analyze this render and return JSON only."
        )
        qa_raw = await asyncio.to_thread(_ask, VISION_SYSTEM, qa_prompt, render_path)
        await send("vision_qa", text=qa_raw)

        try:
            qa = json.loads(qa_raw)
        except Exception:
            qa = {"matches": True, "score": 7, "issues": [], "suggestions": []}

        if qa.get("matches") or qa.get("score", 0) >= 7 or attempt >= MAX_REFINE:
            await send("done", score=qa.get("score", "?"), msg="Generation complete.")
            return

        # ── refine loop ───────────────────────────────────────────────────────
        await send("status", msg=f"Score {qa['score']}/10 — refining based on visual feedback...")
        refine_user = (
            f"Request: {prompt}\n\n"
            f"QA Feedback:\n{qa_raw}\n\n"
            f"Previous script:\n{current_script}"
        )
        current_script = await asyncio.to_thread(_ask, REFINE_SYSTEM, refine_user, render_path)
        if "```" in current_script:
            lines = current_script.split("\n")
            current_script = "\n".join(l for l in lines if not l.strip().startswith("```"))
        await send("script", code=current_script)
        render_path = RENDER_DIR / f"render_{int(time.time()*1000)}.png"

    await send("done", msg="Finished.")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Blender AI Studio")

# session state: last script + render per connection
_sessions: dict[str, dict] = {}

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_UI

@app.websocket("/ws/{sid}")
async def ws_endpoint(ws: WebSocket, sid: str):
    await ws.accept()
    if sid not in _sessions:
        _sessions[sid] = {"script": None, "render": None}
    sess = _sessions[sid]

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            action = msg.get("action")
            prompt = msg.get("prompt", "").strip()

            if not prompt:
                await ws.send_text(json.dumps({"event": "error", "msg": "Empty prompt."}))
                continue

            if action == "generate":
                sess["script"] = None
                sess["render"] = None
                await run_generation(prompt, ws)
                # save last state
                # (run_generation sends "script" events — capture via ws proxy is impractical;
                #  we re-run a script extraction after instead)

            elif action == "edit":
                await run_generation(
                    prompt, ws,
                    edit_script=sess.get("script"),
                    edit_render=Path(sess["render"]) if sess.get("render") else None,
                )

            elif action == "save_state":
                sess["script"] = msg.get("script")
                sess["render"] = msg.get("render_path")

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"event": "error", "msg": str(e)}))
        except Exception:
            pass

# ── Embedded HTML UI ──────────────────────────────────────────────────────────
HTML_UI = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Blender AI Studio</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#0d0d0d;color:#e0e0e0;font-family:'Segoe UI',monospace;height:100vh;display:flex;flex-direction:column;overflow:hidden}
header{background:#111;border-bottom:1px solid #2a2a2a;padding:10px 18px;display:flex;align-items:center;gap:12px}
header h1{font-size:1.1rem;color:#4fc3f7;letter-spacing:1px}
header span{font-size:.75rem;color:#555}
#badge{font-size:.7rem;padding:2px 8px;border-radius:10px;background:#1a1a2e;color:#7986cb;border:1px solid #3f51b5}
.main{display:flex;flex:1;overflow:hidden}
/* ── left panel ── */
#left{width:340px;min-width:240px;display:flex;flex-direction:column;border-right:1px solid #1e1e1e;background:#0e0e0e}
#log{flex:1;overflow-y:auto;padding:10px;font-size:.78rem;line-height:1.6;color:#aaa}
#log .msg{margin-bottom:6px;padding:4px 8px;border-left:3px solid #333;border-radius:2px}
#log .msg.status{border-color:#4fc3f7;color:#b3e5fc}
#log .msg.script{border-color:#7986cb;color:#c5cae9}
#log .msg.vision{border-color:#81c784;color:#c8e6c9}
#log .msg.error{border-color:#ef5350;color:#ffcdd2}
#log .msg.done{border-color:#ffb74d;color:#ffe0b2}
#log .msg.blender{border-color:#444;color:#666;font-size:.7rem;white-space:pre-wrap;word-break:break-all}
/* ── right panel ── */
#right{flex:1;display:flex;flex-direction:column;overflow:hidden}
#viewport{flex:1;display:flex;align-items:center;justify-content:center;background:#080808;position:relative}
#render-img{max-width:100%;max-height:100%;border-radius:4px;box-shadow:0 0 40px #0004}
#placeholder{color:#333;font-size:1rem;text-align:center;user-select:none}
#spinner{display:none;position:absolute;width:48px;height:48px;border:4px solid #222;border-top-color:#4fc3f7;border-radius:50%;animation:spin 0.8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
/* ── script viewer ── */
#script-wrap{height:180px;background:#080f1a;border-top:1px solid #1a2a3a;overflow:hidden;display:flex;flex-direction:column}
#script-header{padding:4px 10px;font-size:.7rem;color:#4fc3f7;background:#0a1520;border-bottom:1px solid #1a2a3a;display:flex;justify-content:space-between;align-items:center;cursor:pointer;user-select:none}
#script-body{flex:1;overflow-y:auto;padding:8px}
#script-code{font-family:monospace;font-size:.72rem;color:#90a4ae;white-space:pre;line-height:1.5}
/* ── bottom bar ── */
#bottom{background:#111;border-top:1px solid #1e1e1e;padding:10px 14px;display:flex;gap:8px;align-items:flex-end}
#prompt{flex:1;background:#1a1a1a;border:1px solid #2a2a2a;border-radius:6px;color:#e0e0e0;padding:9px 12px;font-size:.9rem;resize:none;height:52px;outline:none;transition:border-color .2s}
#prompt:focus{border-color:#4fc3f7}
.btn{padding:10px 16px;border:none;border-radius:6px;cursor:pointer;font-size:.82rem;font-weight:600;letter-spacing:.5px;transition:all .15s}
#btn-gen{background:#1565c0;color:#fff}
#btn-gen:hover{background:#1976d2}
#btn-edit{background:#1b5e20;color:#fff}
#btn-edit:hover{background:#2e7d32}
#btn-gen:disabled,#btn-edit:disabled{background:#2a2a2a;color:#555;cursor:default}
#qa-bar{height:28px;background:#0a1a0a;border-top:1px solid #1a2a1a;display:flex;align-items:center;padding:0 12px;font-size:.72rem;color:#555;gap:8px}
#score-badge{display:none;background:#1b5e20;color:#a5d6a7;padding:2px 8px;border-radius:10px;font-size:.7rem}
</style>
</head>
<body>
<header>
  <h1>Blender AI Studio</h1>
  <span id="badge">gemini-2.0-flash + blender 3.4</span>
  <span id="conn-status" style="margin-left:auto;font-size:.7rem;color:#555">connecting...</span>
</header>
<div class="main">
  <!-- left: log -->
  <div id="left">
    <div id="log"><div style="color:#333;padding:8px">Session log will appear here.</div></div>
  </div>
  <!-- right: viewport + script -->
  <div id="right">
    <div id="viewport">
      <div id="placeholder">
        Type a description below and click Generate<br>
        <span style="font-size:.8rem;color:#222;margin-top:8px;display:block">
          e.g. "a glowing blue crystal on a dark stone floor"
        </span>
      </div>
      <img id="render-img" src="" style="display:none"/>
      <div id="spinner"></div>
    </div>
    <div id="qa-bar">
      <span id="qa-text">No render yet.</span>
      <span id="score-badge"></span>
    </div>
    <div id="script-wrap">
      <div id="script-header" onclick="toggleScript()">
        <span>Generated Blender Script</span>
        <span id="script-toggle">[collapse]</span>
      </div>
      <div id="script-body">
        <pre id="script-code" style="color:#333">No script yet.</pre>
      </div>
    </div>
  </div>
</div>
<div id="bottom">
  <textarea id="prompt" placeholder="Describe your 3D scene... (Enter to generate, Shift+Enter for newline)"></textarea>
  <button class="btn" id="btn-gen" onclick="doGenerate()">Generate</button>
  <button class="btn" id="btn-edit" onclick="doEdit()">Edit</button>
</div>

<script>
const sid = Math.random().toString(36).slice(2);
let ws, busy=false, lastScript=null, lastRenderPath=null, scriptVisible=true;

function connect(){
  ws = new WebSocket(`ws://${location.host}/ws/${sid}`);
  ws.onopen = () => setStatus("connected","#4fc3f7");
  ws.onclose= () => { setStatus("disconnected","#ef5350"); setTimeout(connect,3000); };
  ws.onerror= () => setStatus("error","#ef5350");
  ws.onmessage = e => handle(JSON.parse(e.data));
}

function setStatus(t,c){
  const el=document.getElementById("conn-status");
  el.textContent=t; el.style.color=c;
}

function addLog(cls, html){
  const log=document.getElementById("log");
  const d=document.createElement("div");
  d.className="msg "+cls;
  d.innerHTML=html;
  log.appendChild(d);
  log.scrollTop=log.scrollHeight;
}

function handle(msg){
  switch(msg.event){
    case "status":
      setSpinner(true);
      addLog("status","<b>Status:</b> "+esc(msg.msg));
      break;
    case "script":
      lastScript=msg.code;
      document.getElementById("script-code").textContent=msg.code;
      addLog("script","<b>Script generated</b> ("+msg.code.split("\n").length+" lines)");
      saveState();
      break;
    case "render":
      setSpinner(false);
      lastRenderPath=msg.path;
      showRender(msg.image);
      saveState();
      break;
    case "log":
      addLog("blender",esc(msg.text));
      break;
    case "vision_qa":
      try{
        const qa=JSON.parse(msg.text);
        const score=qa.score||"?";
        const issues=(qa.issues||[]).join("; ")||"None";
        const sb=document.getElementById("score-badge");
        sb.textContent="Score: "+score+"/10"; sb.style.display="inline";
        sb.style.background=score>=7?"#1b5e20":"#b71c1c";
        document.getElementById("qa-text").textContent=
          "AI Visual QA — Issues: "+issues;
        addLog("vision","<b>Vision QA</b>: score="+score+" | "+esc(issues));
      } catch(e){
        addLog("vision","<b>Vision QA</b>: "+esc(msg.text));
      }
      break;
    case "error":
      setSpinner(false); setBusy(false);
      addLog("error","<b>Error:</b> "+esc(msg.msg));
      break;
    case "done":
      setSpinner(false); setBusy(false);
      addLog("done","<b>Done.</b> "+esc(msg.msg||""));
      break;
  }
}

function showRender(b64){
  const img=document.getElementById("render-img");
  const ph=document.getElementById("placeholder");
  img.src=b64;
  img.style.display="block";
  ph.style.display="none";
}

function setSpinner(on){
  document.getElementById("spinner").style.display=on?"block":"none";
}

function setBusy(b){
  busy=b;
  document.getElementById("btn-gen").disabled=b;
  document.getElementById("btn-edit").disabled=b;
}

function saveState(){
  if(ws && ws.readyState===1 && lastScript){
    ws.send(JSON.stringify({
      action:"save_state",
      script:lastScript,
      render_path:lastRenderPath||""
    }));
  }
}

function doGenerate(){
  const p=document.getElementById("prompt").value.trim();
  if(!p||busy) return;
  setBusy(true);
  addLog("status","<b>Generate:</b> "+esc(p));
  ws.send(JSON.stringify({action:"generate",prompt:p}));
}

function doEdit(){
  const p=document.getElementById("prompt").value.trim();
  if(!p||busy) return;
  if(!lastScript){ addLog("error","No scene to edit yet — generate one first."); return; }
  setBusy(true);
  addLog("status","<b>Edit:</b> "+esc(p));
  ws.send(JSON.stringify({action:"edit",prompt:p}));
}

function toggleScript(){
  scriptVisible=!scriptVisible;
  document.getElementById("script-body").style.display=scriptVisible?"flex":"none";
  document.getElementById("script-toggle").textContent=scriptVisible?"[collapse]":"[expand]";
}

function esc(s){ return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;"); }

document.getElementById("prompt").addEventListener("keydown",e=>{
  if(e.key==="Enter"&&!e.shiftKey){ e.preventDefault(); doGenerate(); }
});

connect();
</script>
</body>
</html>
"""

# ── entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Blender AI Studio")
    print("  http://localhost:8082")
    print("  Blender:", BLENDER_BIN)
    print("  Workspace:", WORKSPACE)
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="warning")
