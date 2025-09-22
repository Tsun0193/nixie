from typing import List, Dict, Any
import threading

class SessionStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._rows: list[dict[str, Any]] = []
        self._last_headers: list[str] | None = None

    def clear(self):
        with self._lock:
            self._rows.clear()
            self._last_headers = None

    def add_rows(self, headers: list[str], rows_2d: list[list[str]]):
        with self._lock:
            self._last_headers = headers
            for r in rows_2d:
                self._rows.append(dict(zip(headers, r)))

    def extend_dicts(self, dicts: list[Dict[str, Any]]):
        with self._lock:
            self._rows.extend(dicts)
            if dicts and not self._last_headers:
                self._last_headers = list(dicts[0].keys())

    def get_rows(self) -> list[Dict[str, Any]]:
        with self._lock:
            return list(self._rows)

    def last_headers(self) -> list[str] | None:
        with self._lock:
            return list(self._last_headers) if self._last_headers else None

# Two isolated stores
main_store = SessionStore()   # for "/"
ecus_store = SessionStore()   # for "/ecus"