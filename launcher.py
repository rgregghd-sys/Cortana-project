import os
import sys
import subprocess

def start_aura():
    # This tells the app to find where the 'app.py' is hidden inside the bundle
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    app_path = os.path.join(base_path, "app.py")

    # Starts the Streamlit server automatically
    subprocess.run(["python", "-m", "streamlit", "run", app_path, "--server.headless", "true"])

if __name__ == "__main__":
    start_aura()
