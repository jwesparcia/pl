import urllib.request
import json
import time
import subprocess
import os
import signal

# start server
proc = subprocess.Popen(["python", "e:/moviereco/backend/app.py"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
time.sleep(4)

req = urllib.request.Request("http://localhost:5000/recommend-by-movie", data=json.dumps({"movie_title": "Toy Story"}).encode('utf-8'), headers={'Content-Type': 'application/json'}, method='POST')

try:
    with urllib.request.urlopen(req) as response:
        print("Status:", response.status)
        print("Response:", response.read().decode('utf-8')[:200] + "...")
except Exception as e:
    print("Error:", e)

# stop server
proc.terminate()
proc.kill()

print("\n--- SERVER LOGS ---")
print(proc.stdout.read())
