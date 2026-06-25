import sys
import threading
import time
import webbrowser
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import uvicorn

HOST = "127.0.0.1"
PORT = 8000


def _open():
    time.sleep(1.5)
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    threading.Thread(target=_open, daemon=True).start()
    uvicorn.run("picocal_explorer.app:app", host=HOST, port=PORT)
