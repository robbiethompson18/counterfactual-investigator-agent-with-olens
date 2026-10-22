"""Weirdness-ranking dashboard: python server.py [port]. Serves static/index.html and a small JSON API (re-reads run/ on each request)."""

import json
import socket
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import export

HERE = Path(__file__).parent
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8807
HOSTS = sys.argv[2:] or ["100.86.34.70", "fd7a:115c:a1e0::e934:2247"]


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(HERE / "static"), **k)

    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/ranking":
            return self.send_json(export.build())
        if path.startswith("/api/item/"):
            iid = unquote(path[len("/api/item/") :])
            f = HERE / "run" / "transcripts" / (iid.replace(":", "__") + ".json")
            if "/" in iid or not f.exists():
                return self.send_json({"error": "not found"}, 404)
            d = json.loads(f.read_text())
            items = {it["id"]: it for it in json.loads((HERE / "run" / "items.json").read_text())}
            ms = export.matches().get(iid, [])
            for m in ms:
                o = items.get(m["opponent"], {})
                m["opponent_question"] = o.get("question")
            d["matches"] = sorted(ms, key=lambda m: m["round"])
            return self.send_json(d)
        return super().do_GET()


class S6(ThreadingHTTPServer):
    address_family = socket.AF_INET6


servers = []
for h in HOSTS:
    cls = S6 if ":" in h else ThreadingHTTPServer
    servers.append(cls((h, PORT), H))
for s in servers[1:]:
    threading.Thread(target=s.serve_forever, daemon=True).start()
print(f"serving on {HOSTS} :{PORT}", flush=True)
servers[0].serve_forever()
