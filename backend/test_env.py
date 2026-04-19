import sys
import os
print("Python version:", sys.version)
print("Current directory:", os.getcwd())
try:
    import requests
    print("requests imported")
except ImportError:
    print("requests NOT found")
