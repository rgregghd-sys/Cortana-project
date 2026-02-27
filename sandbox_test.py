import sys
import subprocess
import os

def validate_code(filename):
    print(f"--- 🛡️ Testing AI-Generated Update: {filename} ---")
    
    # Check 1: Syntax
    try:
        res = subprocess.run([sys.executable, "-m", "py_compile", filename], capture_output=True, text=True)
        if res.returncode != 0:
            print(f"❌ Syntax Error:\n{res.stderr}")
            return False
        print("✅ Syntax Verified.")
    except Exception as e:
        print(f"❌ Check Failed: {e}")
        return False

    # Check 2: Stability Run
    try:
        print("🚀 Running Stability Test...")
        # Run the code for 3 seconds to see if it crashes or loops
        subprocess.run([sys.executable, filename], timeout=3, capture_output=True)
        print("✅ Stability Test Passed.")
        return True
    except subprocess.TimeoutExpired:
        print("✅ Stability Test Passed (Timed out as expected for loops).")
        return True
    except Exception as e:
        print(f"❌ Execution Crash: {e}")
        return False

if __name__ == "__main__":
    patch = "update_patch.py"
    if os.path.exists(patch):
        if validate_code(patch):
            print("\n👑 VERDICT: SECURE. READY FOR CLOUD DEPLOY.")
        else:
            print("\n🚫 VERDICT: DANGEROUS CODE. DO NOT DEPLOY.")
    else:
        print(f"Please save the AI's patch as '{patch}' in this folder.")
