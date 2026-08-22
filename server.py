"""Tiny stdlib server for the scorecard explainer.

Runs the whole build once, caches it, and hands the front end a single JSON
payload. Everything interactive on the page (scoring an applicant, moving the
cutoff) is computed in the browser from that payload — no user input ever
reaches the model code.
"""

import json
import os
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import scorecard

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("PORT", 8000))


@lru_cache(maxsize=1)
def cached_analysis() -> str:
    return json.dumps(scorecard.analyze())


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        payload = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            with open(os.path.join(HERE, "index.html"), "rb") as fh:
                self._send(200, fh.read(), "text/html; charset=utf-8")
        elif path == "/api/analyze":
            try:
                self._send(200, cached_analysis())
            except Exception as exc:                    # pragma: no cover
                self._send(500, json.dumps({"error": str(exc)}))
        elif path == "/api/health":
            self._send(200, json.dumps({"ok": True}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    print(f"Building the scorecard (first request takes ~10s)...")
    print(f"  http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
