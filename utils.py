from datetime import datetime

def now_local_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")