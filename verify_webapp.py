import subprocess
import time
import requests
import sys
import os

def check_server():
    print("Starting Spercer Server...")
    # Start the server in a separate process
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server:app", "--port", "8001"],
        cwd=os.getcwd(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True # Capture as string
    )
    
    try:
        # Give it time to boot (increased for heavy imports)
        print("Waiting for server boot (15s)...")
        time.sleep(15)
        
        # 1. Check Frontend Access
        try:
            r = requests.get("http://127.0.0.1:8001/")
            if r.status_code == 200:
                print("Frontend accessible (HTTP 200)")
            else:
                print(f"Frontend failed: {r.status_code}")
        except Exception as e:
            print(f"Connection failed: {e}")
            # Check logs regardless
            outs, errs = process.communicate(timeout=1) # This kills it if not killed? No, communicate waits.
            # actually communicate waits for process to terminate if input is None? 
            # No, communicate reads until EOF. If process is running, it hangs.
            # We should not use communicate on running process without input.
            # Just ignore logs here, we read them at finally.
            return

        # 2. Check API Endpoint (Simulated Query)
        print("Testing API Endpoint (API Mode)...")
        payload = {
            "query": "whats the Torque for brake caliper anchor plate bolts",
            "mode": "API"
        }
        try:
            r = requests.post("http://127.0.0.1:8001/api/query", json=payload)
            if r.status_code == 200:
                data = r.json()
                print("API Query Successful!")
                print(f"   Response Preview: {str(data)[:100]}...")
            else:
                print(f"API Failed: {r.status_code} - {r.text}")
        except Exception as e:
            print(f"API Request Error: {e}")

    finally:
        print("Stopping server process...")
        process.terminate()
        try:
            outs, errs = process.communicate(timeout=5)
            print(f"--- SERVER LOGS ---\n{outs}\n{errs}")
        except:
            print("Could not retrieve logs.")

if __name__ == "__main__":
    check_server()
