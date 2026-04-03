import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.getcwd(), 'backend', '.env'))

from llm_utils import scout_poster_paths_batch

titles = ['Get Shorty (1995)', 'Copycat (1995)', 'Assassins (1995)', 'Powder (1995)', 'Leaving Las Vegas (1995)']
print(f"Scouting for: {titles}")
result = scout_poster_paths_batch(titles)
print(f"Result: {result}")
