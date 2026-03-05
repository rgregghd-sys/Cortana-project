import streamlit as st
from streamlit_mic_recorder import mic_recorder
import threading
import queue
import time
import os
import json
import subprocess
import re
import io
import google.generativeai as genai
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

# --- 1. CONFIGURATION ---
# Replace with your actual key or use streamlit secrets
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "YOUR_API_KEY_HERE")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

MEMORY_FILE = "jarvis_memory.json"
DEV_WORKSPACE = "agent_workspace"
memory_lock = threading.Lock()
task_queue = queue.Queue() # Web/Vision (Agents 0, 1, 2)
dev_queue = queue.Queue()  # R&D (Agents 3, 4)

if not os.path.exists(DEV_WORKSPACE):
    os.makedirs(DEV_WORKSPACE)

# --- 2. PERSISTENT MEMORY & ETHICS ---
def load_mem():
    default = {
        "short_term": [], 
        "long_term": ["System Initialized 2026."], 
        "ethics_filter": "Strict: Explain reasoning for refusals.",
        "learnings": [], 
        "visual_insights": [],
        "emotional_state": "Neutral"
    }
    if not os.path.exists(MEMORY_FILE): return default
    with memory_lock:
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except: return default

def save_mem(data):
    with memory_lock:
        with open(MEMORY_FILE, "w") as f:
            json.dump(data, f, indent=4)

# --- 3. RECURSIVE R&D WING (Agents 3 & 4) ---
def run_code_sandbox(code, agent_id):
    file_path = f"{DEV_WORKSPACE}/agent_{agent_id}_runtime.py"
    with open(file_path, "w") as f:
        f.write(code)
    try:
        # Running in a subprocess for isolation
        res = subprocess.run(["python3", file_path], capture_output=True, text=True, timeout=15)
        return (res.returncode == 0, res.stdout if res.returncode == 0 else res.stderr)
    except Exception as e:
        return (False, str(e))

def developer_swarm_logic():
    """Agent 3 Creates | Agent 4 Reviews and Fixes"""
    while True:
        try:
            mission = dev_queue.get(timeout=10)
            # Agent 3: Initial Draft
            prompt3 = f"Agent 3 Mission: {mission}. Output ONLY raw Python code. No markdown."
            code3 = re.sub(r'```python|```', '', model.generate_content(prompt3).text).strip()
            
            success, output = run_code_sandbox(code3, 3)
            
            if not success:
                # Agent 4: Self-Correction Loop
                prompt4 = f"Agent 4 Review: Agent 3 failed mission '{mission}'.\nCode: {code3}\nError: {output}\nFix the code and output ONLY raw Python."
                code4 = re.sub(r'```python|```', '', model.generate_content(prompt4).text).strip()
                success, output = run_code_sandbox(code4, 4)
                final_status = "Fixed by Agent 4" if success else "Failed Post-Review"
            else:
                final_status = "Success (Agent 3)"

            mem = load_mem()
            mem["learnings"].append(f"[{time.strftime('%H:%M')}] {final_status}: {mission}")
            save_mem(mem)
            dev_queue.task_done()
        except: continue

# --- 4. CORTANA VISION WORKER (Agent 0) ---
def vision_worker():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    service = Service(executable_path='/usr/bin/chromedriver')
    try:
        driver = webdriver.Chrome(service=service, options=options)
        while True:
            try:
                task = task_queue.get(timeout=10)
                url = task if task.startswith("http") else f"https://www.google.com/search?q={task.replace(' ', '+')}"
                driver.get(url)
                time.sleep(2)
                
                # Capture and Analyze
                screenshot = driver.get_screenshot_as_png()
                img = Image.open(io.BytesIO(screenshot))
                analysis = model.generate_content(["Describe the layout and key data on this page:", img]).text
                
                mem = load_mem()
                mem["visual_insights"].append(f"Seen on {driver.title}: {analysis[:150]}...")
                save_mem(mem)
                task_queue.task_done()
            except: continue
    except: pass

# Start background threads
if "swarm_ready" not in st.session_state:
    threading.Thread(target=developer_swarm_logic, daemon=True).start()
    threading.Thread(target=vision_worker, daemon=True).start()
    st.session_state.swarm_ready = True

# --- 5. UI DASHBOARD ---
st.set_page_config(page_title="JARVIS 2026", layout="wide", page_icon="🦾")
mem = load_mem()

# Sidebar: Sensory & Emotional Data
with st.sidebar:
    st.title("🛰️ System Status")
    st.metric("User Emotion", mem["emotional_state"])
    st.divider()
    
    st.subheader("🎙️ Voice Input")
    audio = mic_recorder(start_prompt="Speak Command", stop_prompt="Analyze", key='vox')
    if audio:
        # Process voice for emotion and text (Simplified for full script)
        st.info("Audio received. Processing tone...")

    st.divider()
    st.subheader("👁️ Cortana's Vision")
    for v in reversed(mem["visual_insights"][-3:]):
        st.caption(v)

# Main Body: Chat and Live Lab
chat_col, lab_col = st.columns([1, 1])

with chat_col:
    st.title("🦾 Jarvis Mainframe")
    if "messages" not in st.session_state: st.session_state.messages = []
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])

    if prompt := st.chat_input("Directives..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): st.markdown(prompt)
        
        # Jarvis Decision Logic
        full_ctx = f"MEM: {mem['long_term'][-2:]}\nMOOD: {mem['emotional_state']}"
        res = model.generate_content(f"{full_ctx}\nUSER: {prompt}\nTags: [CODE_TASK: 'mission'] [SEARCH: 'query']")
        
        # Dispatch Tasks
        for ct in re.findall(r'\[CODE_TASK:\s*(.*?)\]', res.text): dev_queue.put(ct)
        for st_task in re.findall(r'\[SEARCH:\s*(.*?)\]', res.text): task_queue.put(st_task)
        
        reply = re.sub(r'\[.*?\]', '', res.text)
        with st.chat_message("assistant"): st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

with lab_col:
    st.title("🧪 R&D Glass Box")
    tabs = st.tabs(["Agent 3 (Draft)", "Agent 4 (Correction)", "Evolution Logs"])
    
    with tabs[0]:
        path3 = f"{DEV_WORKSPACE}/agent_3_runtime.py"
        if os.path.exists(path3):
            with open(path3, "r") as f: st.code(f.read(), language="python")
    with tabs[1]:
        path4 = f"{DEV_WORKSPACE}/agent_4_runtime.py"
        if os.path.exists(path4):
            with open(path4, "r") as f: st.code(f.read(), language="python")
    with tabs[2]:
        for l in reversed(mem["learnings"][-10:]):
            st.write
