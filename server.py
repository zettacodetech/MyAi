"""
MyAI backend - 0 dan yozilgan neyron tarmoqni web orqali xizmat qiladi.
Tashqi framework yo'q, faqat Python stdlib (http.server).
"""
import json
import re
import os
import time
import datetime
import urllib.request
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from model import MyAI

try:
    import anthropic
    _ANTHROPIC_OK = True
except Exception:
    _ANTHROPIC_OK = False
from transformer import MiniGPT

ROOT = Path(__file__).parent
PUBLIC = ROOT / "public"
MODEL_PATH = ROOT / "model.pkl"

GPT_PATH = ROOT / "gpt.pkl"
print("Model yuklanmoqda...")
try:
    AI = MiniGPT.load(GPT_PATH)
    print(f"Transformer yuklandi (vocab={AI.vocab_size})")
except Exception as e:
    AI = MyAI.load(MODEL_PATH)
    print(f"MLP model yuklandi (Transformer topilmadi: {e})")


def load_env():
    """.env fayldan kalitlarni o'qiydi."""
    env = {}
    ef = ROOT / ".env"
    if ef.exists():
        for line in ef.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env

def load_system_prompt():
    """system_prompt.txt ni o'qib, {current_date} ni bugungi sanaga almashtiradi."""
    f = ROOT / "system_prompt.txt"
    default = "Sen MyAI'san — o'zbek foydalanuvchilar uchun yordamchi. O'zbek tilida tabiiy javob ber."
    if not f.exists():
        return default
    today = datetime.date.today().strftime("%Y-%m-%d")
    return f.read_text(encoding="utf-8").replace("{current_date}", today)


ENV = load_env()
SERPAPI_KEY = os.environ.get("SERPAPI_KEY") or ENV.get("SERPAPI_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY") or ENV.get("ANTHROPIC_API_KEY", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY") or ENV.get("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-flash-latest"
if GEMINI_KEY:
    print("Gemini rejimi yoqilgan")
if ANTHROPIC_KEY and _ANTHROPIC_OK:
    print("Claude rejimi yoqilgan (anthropic SDK)")
if SERPAPI_KEY:
    print("Web qidiruv yoqilgan (SerpApi)")


def web_search(query):
    """SerpApi orqali Google'da qidiradi va eng yaxshi javobni qaytaradi."""
    if not SERPAPI_KEY:
        return {"answer": "Web qidiruv sozlanmagan (.env da SERPAPI_KEY yo'q).", "source": "", "link": ""}
    params = urllib.parse.urlencode({
        "engine": "google", "q": query, "hl": "uz", "gl": "uz",
        "num": "5", "api_key": SERPAPI_KEY,
    })
    url = f"https://serpapi.com/search?{params}"
    try:
        with urllib.request.urlopen(url, timeout=45) as r:
            d = json.loads(r.read())
    except Exception as e:
        return {"answer": f"Qidiruvda xato: {e}", "source": "", "link": ""}

    # Ustuvorlik: answer_box > knowledge_graph > ai_overview > organic
    ab = d.get("answer_box") or {}
    if ab:
        ans = ab.get("answer") or ab.get("snippet") or ab.get("result") or ""
        if ans:
            return {"answer": ans, "source": ab.get("title", "Google"),
                    "link": ab.get("link", "")}
    kg = d.get("knowledge_graph") or {}
    if kg.get("description"):
        return {"answer": kg["description"], "source": kg.get("title", "Google"),
                "link": kg.get("source", {}).get("link", "") if isinstance(kg.get("source"), dict) else ""}
    ai = d.get("ai_overview") or {}
    blocks = ai.get("text_blocks") or []
    if blocks:
        texts = []
        for b in blocks:
            if b.get("snippet"):
                texts.append(b["snippet"])
            for it in (b.get("list") or []):
                if it.get("snippet"):
                    texts.append("• " + it["snippet"])
        if texts:
            return {"answer": " ".join(texts)[:800], "source": "Google AI Overview", "link": ""}
    org = d.get("organic_results") or []
    if org:
        top = org[0]
        return {"answer": top.get("snippet") or top.get("title", ""),
                "source": top.get("title", "Google"), "link": top.get("link", "")}
    return {"answer": "Hech narsa topilmadi.", "source": "", "link": ""}


def _serp(params):
    params = dict(params); params["api_key"] = SERPAPI_KEY
    url = "https://serpapi.com/search?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=45) as r:
        return json.loads(r.read())


def news_search(query):
    """Google yangiliklaridan qidiradi."""
    if not SERPAPI_KEY:
        return {"type": "news", "items": []}
    try:
        d = _serp({"engine": "google_news", "q": query, "hl": "uz", "gl": "uz"})
    except Exception as e:
        return {"type": "news", "items": [], "error": str(e)}
    items = []
    for n in (d.get("news_results") or [])[:5]:
        # ba'zan ichki 'stories' bo'ladi
        if n.get("stories"):
            n = n["stories"][0]
        items.append({
            "title": n.get("title", ""),
            "link": n.get("link", ""),
            "source": (n.get("source") or {}).get("name", "") if isinstance(n.get("source"), dict) else (n.get("source") or ""),
            "date": n.get("date", ""),
        })
    return {"type": "news", "items": items}


def image_search(query):
    """Google rasmlaridan qidiradi."""
    if not SERPAPI_KEY:
        return {"type": "images", "items": []}
    try:
        d = _serp({"engine": "google_images", "q": query, "hl": "uz", "gl": "uz"})
    except Exception as e:
        return {"type": "images", "items": [], "error": str(e)}
    items = []
    for im in (d.get("images_results") or [])[:8]:
        items.append({
            "thumb": im.get("thumbnail", ""),
            "original": im.get("original", im.get("thumbnail", "")),
            "title": im.get("title", ""),
            "link": im.get("link", ""),
        })
    return {"type": "images", "items": items}


FACT_WORDS = ("qancha", "necha", "nechta", "qachon", "qayer", "qaysi",
              "poytaxti", "aholisi", "narxi", "ob-havo", "obhavo", "nechanchi",
              "qanaqa", "nima uchun", "kim ", "qanday qilib", "eng katta",
              "eng baland", "eng uzun", "necha yil")
CHITCHAT = ("salom", "assalom", "qalaysiz", "qandaysiz", "rahmat", "xayr",
            "ismingiz", "isming", "kimsan", "kim san", "yaxshimisiz",
            "tanishaylik", "qalay", "nima gap")


def is_factual(q):
    """Savol faktik (internetdan qidirish kerak)mi yoki oddiy suhbatmi?"""
    ql = q.lower()
    if any(w in ql for w in CHITCHAT):
        return False
    if "?" in q:
        return True
    if any(w in ql for w in FACT_WORDS):
        return True
    if any(ch.isdigit() for ch in q):
        return True
    return False


# Oddiy suhbat uchun tayyor, sifatli o'zbekcha javoblar
INTENTS = [
    (("assalom", "salom", "salomat"),
     "Vaalaykum assalom! Men MyAI. Sizga qanday yordam bera olaman? 😊"),
    (("qalaysiz", "qandaysiz", "yaxshimisiz", "qalay", "nima gap", "ishlar qalay"),
     "Rahmat, men yaxshiman! O'zingiz qalaysiz? Biror narsa so'rang — yordam beraman."),
    (("ismingiz", "isming", "oting", "kimsan", "kim san", "o'zing kim", "sen kimsan"),
     "Mening ismim MyAI. Men Inomjon 0 dan qurgan sun'iy intellektman. Savollarga javob beraman va internetdan ma'lumot topaman."),
    (("rahmat", "tashakkur", "raxmat", "minnatdor"),
     "Arzimaydi! Doim xizmatdaman. Yana savolingiz bo'lsa, bemalol yozing. 🙌"),
    (("xayr", "ko'rishguncha", "korishguncha", "salomat bo'ling"),
     "Xayr! Yana keling, sizni kutaman. 👋"),
    (("nima qila olasan", "yordam ber", "nima qilasan", "vazifang", "nima ish qilasan"),
     "Men shularni qila olaman:\n• Savollaringizga javob beraman\n• Internetdan (Google) ma'lumot topaman\n• Yangiliklar va rasmlarni qidiraman\nSinab ko'ring — masalan: 'Toshkent aholisi qancha?'"),
    (("zo'r", "ajoyib", "yaxshi ishlayapsan", "barakalla", "qoyil"),
     "Rahmat! Xursandman. Yana biror narsa so'rang. 😊"),
    (("yosh", "necha yosh", "nechchi yosh"),
     "Men dasturman, yoshim yo'q 😄 Lekin har kuni yangi narsa o'rganaman."),
]


def match_intent(q):
    """Oddiy suhbat bo'lsa tayyor javob qaytaradi, aks holda None."""
    ql = q.lower().strip("?!. ")
    for keys, reply in INTENTS:
        for k in keys:
            if k in ql:
                return reply
    return None


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
        elif self.path == "/api/search":
            data = self._read_body()
            query = (data.get("query") or data.get("prompt") or "").strip()
            if not query:
                self._send(400, json.dumps({"error": "query bo'sh"}))
                return
            result = web_search(query)
            self._send(200, json.dumps(result, ensure_ascii=False))
        elif self.path == "/api/gemini":
            self.stream_gemini()
        elif self.path == "/api/claude":
            self.stream_claude()
        elif self.path == "/api/news":
            data = self._read_body(); q = (data.get("query") or data.get("prompt") or "").strip()
            self._send(200, json.dumps(news_search(q), ensure_ascii=False))
        elif self.path == "/api/images":
            data = self._read_body(); q = (data.get("query") or data.get("prompt") or "").strip()
            self._send(200, json.dumps(image_search(q), ensure_ascii=False))
        elif self.path == "/api/chat":
            # AQLLI rejim: o'zi qaror qiladi - qidirish yoki neyron tarmoq
            data = self._read_body()
            q = (data.get("prompt") or "").strip()
            if not q:
                self._send(400, json.dumps({"error": "bo'sh"}))
                return
            ql = q.lower()
            _intent = match_intent(q)
            if _intent:
                self._send(200, json.dumps({"answer": _intent, "mode": "ai"}, ensure_ascii=False))
            elif SERPAPI_KEY and any(w in ql for w in ("rasm", "surat", "foto", "rasmini", "suratini")):
                res = image_search(q); res["mode"] = "images"
                self._send(200, json.dumps(res, ensure_ascii=False))
            elif SERPAPI_KEY and any(w in ql for w in ("yangilik", "yangiliklar", "xabar", "news")):
                res = news_search(q); res["mode"] = "news"
                self._send(200, json.dumps(res, ensure_ascii=False))
            elif SERPAPI_KEY and is_factual(q):
                res = web_search(q)
                res["mode"] = "web"
                self._send(200, json.dumps(res, ensure_ascii=False))
            elif SERPAPI_KEY:
                # noma'lum bo'lsa ham internetdan qidirib ko'ramiz
                res = web_search(q)
                res["mode"] = "web"
                self._send(200, json.dumps(res, ensure_ascii=False))
            else:
                self._send(200, json.dumps({
                    "answer": "Buni to'liq tushunmadim. Biror savol bering (masalan 'Yer nechta sun'iy yo'ldoshi bor?') yoki salomlashing. 😊",
                    "mode": "ai"}, ensure_ascii=False))
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

    def stream_gemini(self):
        """Google Gemini (bepul) orqali streaming javob."""
        data = self._read_body()
        prompt = (data.get("prompt") or "").strip()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        def sse(txt):
            payload = json.dumps({"c": txt}, ensure_ascii=False)
            self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
            self.wfile.flush()

        if not GEMINI_KEY:
            sse("Gemini kaliti yo'q (.env da GEMINI_API_KEY).")
        else:
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{GEMINI_MODEL}:streamGenerateContent?alt=sse")
            body = {
                "systemInstruction": {"parts": [{"text": load_system_prompt()}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            }
            req = urllib.request.Request(
                url, data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json", "x-goog-api-key": GEMINI_KEY},
            )
            last_err = None
            got = False
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        for raw in resp:
                            raw = raw.strip()
                            if not raw or not raw.startswith(b"data:"):
                                continue
                            try:
                                obj = json.loads(raw[5:].strip())
                                parts = obj["candidates"][0]["content"]["parts"]
                                for p in parts:
                                    if p.get("text"):
                                        got = True
                                        sse(p["text"])
                            except Exception:
                                continue
                    break  # muvaffaqiyat
                except urllib.error.HTTPError as e:
                    body = e.read().decode("utf-8", "ignore")
                    if e.code in (503, 429, 500) and attempt < 3 and not got:
                        time.sleep(1.5 * (attempt + 1))  # band - kutib qayta uramiz
                        continue
                    try:
                        last_err = json.loads(body)["error"]["message"][:200]
                    except Exception:
                        last_err = body[:200]
                    sse(f"[Gemini xatosi {e.code}: {last_err}]")
                    break
                except Exception as e:
                    if attempt < 3 and not got:
                        time.sleep(1.5)
                        continue
                    sse(f"[Gemini ulanish xatosi: {e}]")
                    break
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def stream_claude(self):
        """Haqiqiy Claude (Anthropic SDK) orqali streaming javob - skill ko'rsatmasi bo'yicha."""
        data = self._read_body()
        prompt = (data.get("prompt") or "").strip()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        def sse(txt):
            payload = json.dumps({"c": txt}, ensure_ascii=False)
            self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
            self.wfile.flush()

        if not _ANTHROPIC_OK:
            sse("[anthropic SDK o'rnatilmagan]")
        elif not ANTHROPIC_KEY:
            sse("Claude kaliti yo'q. .env ga ANTHROPIC_API_KEY qo'shing (console.anthropic.com).")
        else:
            try:
                client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
                with client.messages.stream(
                    model="claude-opus-4-8",
                    max_tokens=2048,
                    thinking={"type": "adaptive"},
                    system=load_system_prompt(),
                    messages=[{"role": "user", "content": prompt}],
                ) as stream:
                    for text in stream.text_stream:
                        sse(text)
            except Exception as e:
                sse(f"[Claude xatosi: {e}]")
        try:
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
    port = os.environ.get("PORT")
    if port:
        main(int(port))
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 3070)
