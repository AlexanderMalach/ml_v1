import os
from datetime import datetime


class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        full_message = f"[{timestamp}] {message}"
        print(full_message)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(full_message + "\n")
