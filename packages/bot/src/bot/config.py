import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

this_file: Path = Path(__file__)
root_path: Path = this_file.parent

THRESHOLD = float(os.environ.get("THRESHOLD"))
DISCORD_TOKEN = str(os.environ.get("DISCORD_TOKEN"))
API_URL= str(os.environ.get("API_URL"))
