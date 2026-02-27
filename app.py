import streamlit as st
import requests
from streamlit_lottie import st_lottie
import json

# 1. SETUP & THEME
st.set_page_config(page_title="AURA NEURAL LINK", page_icon="💠", layout="centered")

# Custom CSS to make the hologram "pop" against a dark background
st.markdown("""
    <style>
    .stApp { background-color: #050505; }
    h1 { font-family: 'Orbitron', sans-serif; color: #00d4ff; text-shadow: 0px 0px 15px #00d4ff; }
    </style>
    """, unsafe_allow_html=True)

# 2. THE HOLOGRAM LOAD (The 3D Face)
def load_hologram(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Unique Neon Digital Entity Hologram
hologram_url = "https://lottie.host/805e3230-0193-4712-88f1-c67d37704250/S7X7t6G9R1.json"
aura_face = load_hologram(hologram_url)

# 3. HEADER
st.markdown("<h1 style='text-align: center;'>AURA : CORE</h1>", unsafe_allow_html=True)

# 4. RENDER THE 3D FACE
if aura_face:
    st_lottie(
        aura_face,
        speed=1,
        reverse=False,
        loop=True,
        quality="high", # High quality for the 3D effect
        height=350,
        key="aura_hologram"
    )
else:
    st.error("Holographic Interface Offline: Check Connection")

# 5. CHAT SYSTEM
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Command Aura..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # (Logic for your Local Unbiased Brain goes here)
    response = "Neural link established. I am Aura. How shall we proceed?"
    
    with st.chat_message("assistant"):
        st.markdown(response)
        # Voice Output
        st.components.v1.html(f"""
            <script>
                var msg = new SpeechSynthesisUtterance({json.dumps(response)});
                window.speechSynthesis.speak(msg);
            </script>
        """, height=0)
    st.session_state.messages.append({"role": "assistant", "content": response})
