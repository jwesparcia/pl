import requests
import json

url = "http://localhost:5005/recommend-by-movie"
payload = {
    "movie_title": "Four Rooms",
    "user_context": {"mode": "default"}
}

try:
    print(f"Sending request to {url} for 'Avatar'...")
    response = requests.post(url, json=payload, timeout=60)
    if response.status_code == 200:
        data = response.json()
        recos = data.get('recommendations', [])
        print(f"Received {len(recos)} recommendations.")
        for i, r in enumerate(recos):
            print(f"{i+1}. {r['title']} (Score: {r['score']:.3f}, Rating: {r['predicted_rating']})")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Request failed: {e}")
