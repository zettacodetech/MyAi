"""
MyAI backend - 0 dan yozilgan neyron tarmoqni web orqali xizmat qiladi.
Tashqi framework yo'q, faqat Python stdlib (http.server).
"""
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from model import MyAI

ROOT = Path(__file__).parent
PUBLIC = ROOT / "public"
MODEL_PATH = ROOT / "model.pkl"

print("Model yuklanmoqda...")
AI = MyAI.load(MODEL_PATH)
print(f"Model tayyor (vocab={AI.vocab_size})")


def clean_reply(text, prompt):
    """Generatsiyani tozalaydi: promptdan keyingi qismni olib, to'liq jumla(lar) qaytaradi."""
    out = text[len(prompt):] if text.startswith(prompt) else text
    out = out.lstrip(" .?!,\n")   # bosh belgilarni tashlaymiz
    # kamida ~12 belgilik to'liq jumla to'planguncha yig'amiz
    result = ""
    for ch in out:
        result += ch
        if ch in ".?!" and len(result.strip()) >= 12:
            break
    result = result.replace("\n", " ").strip()
    if not result:
        result = out.replace("\n", " ").strip()[:80]
    return result or "..."


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.wfile.write(body)

    def log_message(self, *a):
        pass  # jim

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            path = "/index.html"
        f = PUBLIC / path.lstrip("/")
        if f.is_file() and PUBLIC in f.resolve().parents or f == (PUBLIC / "index.html"):
            ctype = {
                ".html": "text/html", ".css": "text/css",
                ".js": "application/javascript",
            }.get(f.suffix, "text/plain")
            self._send(200, f.read_bytes(), ctype)
        else:
            self._send(404, "Topilmadi", "text/plain")

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            return json.loads(self.rfile.read(length) or "{}")
        except Exception:
            return {}

    def do_POST(self):
        if self.path == "/api/generate":
            data = self._read_body()
            prompt = (data.get("prompt") or "").strip().lower()
            temp = float(data.get("temperature", 0.7))
            seed_text = prompt if prompt else "salom"
            raw = AI.generate(seed_text, n=160, temperature=temp)
            reply = clean_reply(raw, seed_text)
            self._send(200, json.dumps({"response": reply}, ensure_ascii=False))
        elif self.path == "/api/stream":
            self.stream_reply()
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def stream_reply(self):
        """Javobni harf-harf (SSE) yuboradi - jonli chat effekti."""
        data = self._read_body()
        prompt = (data.get("prompt") or "").strip().lower()
        temp = float(data.get("temperature", 0.7))
        seed_text = prompt if prompt else "salom"

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        started = False
        buf = ""
        try:
            for ch in AI.generate_stream(seed_text, n=220, temperature=temp):
                if not started:
                    if ch in " .?!,\n":
                        continue
                    started = True
                buf += ch
                payload = json.dumps({"c": ch}, ensure_ascii=False)
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
                if ch in ".?!" and len(buf.strip()) >= 12:
                    break
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


def main(port=3070):
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"MyAI sayti: http://localhost:{port}")
    srv.serve_forever()


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3070)
