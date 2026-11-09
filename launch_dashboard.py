"""Start the local demonstration and open it in the default browser."""
from pathlib import Path
import subprocess
import time
import urllib.request
import webbrowser

root = Path(__file__).resolve().parent
url = 'http://127.0.0.1:8501'


def is_ready():
    try:
        with urllib.request.urlopen(url + '/_stcore/health', timeout=2) as response:
            return response.read() == b'ok'
    except (OSError, TimeoutError):
        return False


if not is_ready():
    with (root / 'dashboard.log').open('a', encoding='utf-8') as log:
        subprocess.Popen(
            [str(root / '.venv/Scripts/python.exe'), '-m', 'streamlit', 'run',
             str(root / 'app.py'), '--server.headless=true', '--server.address=127.0.0.1', '--server.port=8501'],
            cwd=root, stdout=log, stderr=log,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS,
        )
    for _ in range(30):
        if is_ready():
            break
        time.sleep(1)

if is_ready():
    webbrowser.open(url)
else:
    import ctypes
    ctypes.windll.user32.MessageBoxW(None, 'Unable to start the dashboard. Check dashboard.log in the project folder.', 'Credit Risk Decision Lab', 16)
