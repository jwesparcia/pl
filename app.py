import sys
import os

# Ensure the backend directory is in the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Import the main Flask 'app' object from the backend/app.py folder
# This allows deployment platforms like Render to find the 'app' module
# in the root directory automatically.
from backend.app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5005))
    app.run(host="0.0.0.0", port=port)
