import logging
from logging.handlers import RotatingFileHandler
import os

def setup_logging():
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    # Optional: add rotating file handler
    log_file = os.getenv("LOG_FILE")
    if log_file:
        handler = RotatingFileHandler(log_file, maxBytes=10_000_000, backupCount=3)
        handler.setLevel(log_level)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)