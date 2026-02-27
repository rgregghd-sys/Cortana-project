import streamlit as st
import requests
from streamlit_lottie import st_lottie
from memory import get_all_memory
from emailer import send_patch_to_user # Import our new emailer

st.set_page_config(page_title="Cortana App", page_icon="🌐", layout="centered")

def load_lottieurl(url: str):
    r = requests.get(url)
    return r.json() if r.status_code == 200 else None

# 3D Hologram Face
lottie_face = load_lottieurl("https://lottie.host/809f30b9-50c2-4809-9069-7c859d0473e0/3z8K1nB7Xo.json")

st.markdown("<h1 style='text-align: center; color: #00d4ff;'>CORTANA SYSTEM</h1>", unsafe_allow_html=True)

if lottie_face:
    st_lottie(lottie_face, height=300, key="cortana_face")

# --- COMMAND CENTER ---
st.sidebar.title("🛠️ AI Controls")
if st.sidebar.button("Scan for Improvements"):
    with st.spinner("Analyzing code efficiency..."):
        # We simulate the AI writing a patch
        dummy_patch = "update_patch.py"
        with open(dummy_patch, "w") as f:
            f.write("# Optimized Logic Patch\ndef check_efficiency():\n    return True")
        
        # Call the emailer we just set up
        send_patch_to_user(dummy_patch, "Optimization found in the logic_split node.")
        st.sidebar.success("Patch sent to your email!")

st.markdown("---")
st.subheader("🧠 Digested Intelligence")

memory_data = get_all_memory()
if not memory_data:
    st.info("Searching the web for new data...")
else:
    for info, ts in memory_data[:5]:
        st.write(f"**{ts}**")
        st.info(info)
