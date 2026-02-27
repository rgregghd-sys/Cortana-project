import streamlit as st
import requests
from streamlit_lottie import st_lottie
import time

# 1. SETTING THE IDENTITY
st.set_page_config(page_title="Aura AI Interface", page_icon="💠", layout="centered")

# Eye-catching unique header
st.markdown("<h1 style='text-align: center; color: #00d4ff;'>AURA: Core Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>v2.0 // Neural Link Active</p>", unsafe_allow_html=True)

# 2. THE 3D VISUAL (Lottie Pulse)
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# This is a sleek, neon blue pulsing orb
lottie_aura = load_lottieurl("https://lottie.host/805e3230-0193-4712-88f1-c67d37704250/S7X7t6G9R1.json")

with st.container():
    st_lottie(lottie_aura, height=300, key="aura_visual")

# 3. CONVERSATION INTERFACE
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. CHAT INPUT & VOICE LOGIC
if prompt := st.chat_input("Connect with Aura..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # AI Response placeholder (Link this to your Brain.py later)
    response = f"I am Aura. I am processing your request: '{prompt}'"
    
    with st.chat_message("assistant"):
        st.markdown(response)
        # BROWSER TALK: This makes the browser speak the text
        st.components.v1.html(f"""
            <script>
                var msg = new SpeechSynthesisUtterance('{response}');
                msg.voice = speechSynthesis.getVoices().filter(function(voice) {{ return voice.name == 'Google US English'; }})[0];
                window.speechSynthesis.speak(msg);
            </script>
        """, height=0)

    st.session_state.messages.append({"role": "assistant", "content": response})
