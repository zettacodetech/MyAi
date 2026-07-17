"""
MyAI backend - 0 dan yozilgan neyron tarmoqni web orqali xizmat qiladi.
Tashqi framework yo'q, faqat Python stdlib (http.server).
"""
import json
import re
import os
import time
import datetime
import hashlib
import secrets
import base64
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
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID") or ENV.get("GOOGLE_CLIENT_ID", "")
OCOYA_KEY = os.environ.get("OCOYA_API_KEY") or ENV.get("OCOYA_API_KEY", "")
OCOYA_BASE = "https://www.app.ocoya.com/api/_public/v1"
FIREFLIES_KEY = os.environ.get("FIREFLIES_API_KEY") or ENV.get("FIREFLIES_API_KEY", "")
MEDIASTACK_KEY = os.environ.get("MEDIASTACK_API_KEY") or ENV.get("MEDIASTACK_API_KEY", "")
AISHA_KEY = os.environ.get("AISHA_API_KEY") or ENV.get("AISHA_API_KEY", "")
HF_KEY = os.environ.get("HF_API_KEY") or ENV.get("HF_API_KEY", "")
OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY") or ENV.get("OPENROUTER_KEY", "")
_gem_raw = os.environ.get("GEMINI_API_KEY") or ENV.get("GEMINI_API_KEY", "")
GEMINI_KEYS = [k.strip() for k in _gem_raw.split(",") if k.strip()]
GEMINI_KEY = GEMINI_KEYS[0] if GEMINI_KEYS else ""
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


def research_context(query):
    """SerpApi'dan qidiruv natijalarini (kontekst) yig'adi."""
    if not SERPAPI_KEY:
        return []
    try:
        d = _serp({"engine": "google", "q": query, "hl": "uz", "gl": "uz", "num": "6"})
    except Exception:
        return []
    out = []
    ab = d.get("answer_box") or {}
    if ab.get("snippet") or ab.get("answer"):
        out.append({"title": ab.get("title", "Answer"), "snippet": ab.get("snippet") or ab.get("answer"), "link": ab.get("link", "")})
    for o in (d.get("organic_results") or [])[:6]:
        out.append({"title": o.get("title", ""), "snippet": o.get("snippet", ""), "link": o.get("link", "")})
    return out


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


USERS_FILE = ROOT / "data" / "users.json"


def load_users():
    try:
        return json.loads(USERS_FILE.read_text())
    except Exception:
        return {}


def save_users(u):
    USERS_FILE.parent.mkdir(exist_ok=True)
    USERS_FILE.write_text(json.dumps(u, ensure_ascii=False, indent=2))


def hash_pw(pw, salt):
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), 100000).hex()


def ocoya_req(path, method="GET", body=None):
    url = OCOYA_BASE + path
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"X-API-Key": OCOYA_KEY}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read())


def ocoya_post(caption, scheduled_at=None):
    if not OCOYA_KEY:
        return {"error": "Ocoya kaliti yo'q"}
    try:
        ws = ocoya_req("/workspaces")
        if not ws:
            return {"error": "Ocoya workspace topilmadi"}
        wid = ws[0]["id"]
        payload = {"caption": caption}
        if scheduled_at:
            payload["scheduledAt"] = scheduled_at
        d = ocoya_req("/post?workspaceId=" + wid, "POST", payload)
        return {"ok": True, "postGroupId": d.get("postGroupId"), "workspace": ws[0].get("name", "")}
    except Exception as e:
        return {"error": str(e)}


def aisha_tts(text):
    if not AISHA_KEY:
        return {"error": "Aisha kaliti yo'q"}
    try:
        req = urllib.request.Request("https://back.aisha.group/api/v1/tts/post/",
            data=json.dumps({"transcript": text[:900]}).encode("utf-8"),
            headers={"X-Api-Key": AISHA_KEY, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=45) as r:
            d = json.loads(r.read())
        return {"audio": d.get("audio_path")}
    except Exception as e:
        return {"error": str(e)}


def mediastack_news(limit=8):
    if not MEDIASTACK_KEY:
        return {"error": "Mediastack kaliti yo'q"}
    try:
        url = "http://api.mediastack.com/v1/news?" + urllib.parse.urlencode({
            "access_key": MEDIASTACK_KEY, "limit": limit, "languages": "en,ru", "sort": "published_desc"})
        with urllib.request.urlopen(url, timeout=20) as r:
            d = json.loads(r.read())
        if isinstance(d.get("error"), dict):
            return {"error": d["error"].get("message", "Mediastack xatosi")[:120]}
        items = [{"title": a.get("title"), "source": a.get("source"),
                  "url": a.get("url"), "date": (a.get("published_at") or "")[:10]}
                 for a in (d.get("data") or []) if a.get("title")]
        return {"items": items}
    except Exception as e:
        return {"error": str(e)}


def fireflies_meetings(limit=10):
    if not FIREFLIES_KEY:
        return {"error": "Fireflies kaliti yo'q"}
    q = "{ transcripts(limit: %d) { title date summary { overview } } }" % limit
    try:
        req = urllib.request.Request("https://api.fireflies.ai/graphql",
            data=json.dumps({"query": q}).encode("utf-8"),
            headers={"Authorization": "Bearer " + FIREFLIES_KEY, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=25) as r:
            d = json.loads(r.read())
        ts = (d.get("data") or {}).get("transcripts") or []
        out = []
        for t in ts:
            summ = (t.get("summary") or {}).get("overview") or ""
            out.append({"title": t.get("title", "Yig'ilish"), "date": t.get("date"), "overview": summ})
        return {"meetings": out}
    except Exception as e:
        return {"error": str(e)}


def decode_jwt(token):
    """Google ID token (JWT) payloadini ochadi (imzo tekshirilmaydi, demo uchun)."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload))
    except Exception:
        return None


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
        if self.path.split("?")[0] == "/api/config":
            self._send(200, json.dumps({"googleClientId": GOOGLE_CLIENT_ID}))
            return
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
        if self.path == "/api/register":
            d = self._read_body()
            name = (d.get("name") or "").strip()
            email = (d.get("email") or "").strip().lower()
            pw = d.get("password") or ""
            if not name or not email or len(pw) < 4:
                self._send(400, json.dumps({"error": "Ism, email va kamida 4 belgili parol kerak"}, ensure_ascii=False)); return
            users = load_users()
            if email in users:
                self._send(409, json.dumps({"error": "Bu email allaqachon ro'yxatdan o'tgan"}, ensure_ascii=False)); return
            salt = secrets.token_hex(16)
            token = secrets.token_hex(24)
            users[email] = {"name": name, "salt": salt, "hash": hash_pw(pw, salt), "token": token}
            save_users(users)
            self._send(200, json.dumps({"ok": True, "name": name, "email": email, "token": token}, ensure_ascii=False)); return
        if self.path == "/api/login":
            d = self._read_body()
            email = (d.get("email") or "").strip().lower()
            pw = d.get("password") or ""
            users = load_users()
            u = users.get(email)
            if not u or u["hash"] != hash_pw(pw, u["salt"]):
                self._send(401, json.dumps({"error": "Email yoki parol xato"}, ensure_ascii=False)); return
            self._send(200, json.dumps({"ok": True, "name": u["name"], "email": email, "token": u["token"]}, ensure_ascii=False)); return
        if self.path == "/api/google":
            d = self._read_body()
            info = decode_jwt(d.get("credential") or "")
            if not info or not info.get("email"):
                self._send(400, json.dumps({"error": "Google token yaroqsiz"}, ensure_ascii=False)); return
            email = info["email"].lower()
            name = info.get("name") or email.split("@")[0]
            users = load_users()
            if email in users:
                token = users[email].get("token") or secrets.token_hex(24)
                users[email]["token"] = token; users[email]["name"] = name
            else:
                token = secrets.token_hex(24)
                users[email] = {"name": name, "google": True, "token": token, "picture": info.get("picture", "")}
            save_users(users)
            self._send(200, json.dumps({"ok": True, "name": name, "email": email, "token": token, "picture": info.get("picture", "")}, ensure_ascii=False)); return
        if self.path == "/api/tts":
            d = self._read_body()
            txt = (d.get("text") or "").strip()
            if not txt:
                self._send(400, json.dumps({"error": "matn bo'sh"}, ensure_ascii=False)); return
            self._send(200, json.dumps(aisha_tts(txt), ensure_ascii=False)); return
        if self.path == "/api/worldnews":
            self._send(200, json.dumps(mediastack_news(8), ensure_ascii=False)); return
        if self.path == "/api/fireflies":
            self._send(200, json.dumps(fireflies_meetings(10), ensure_ascii=False)); return
        if self.path == "/api/ocoya":
            d = self._read_body()
            caption = (d.get("caption") or d.get("prompt") or "").strip()
            if not caption:
                self._send(400, json.dumps({"error": "Matn bo'sh"}, ensure_ascii=False)); return
            self._send(200, json.dumps(ocoya_post(caption, d.get("scheduledAt")), ensure_ascii=False)); return
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
        elif self.path == "/api/hf":
            self.stream_hf()
        elif self.path == "/api/openrouter":
            self.stream_openrouter()
        elif self.path == "/api/research":
            self.stream_research()
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

        model_id = (data.get("model") or GEMINI_MODEL)
        if not re.match(r"^[A-Za-z0-9._-]+$", model_id):
            model_id = GEMINI_MODEL
        if not GEMINI_KEY:
            sse("Gemini kaliti yo'q (.env da GEMINI_API_KEY).")
        else:
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{model_id}:streamGenerateContent?alt=sse")
            uparts = [{"text": prompt}]
            img = data.get("image")
            if img and img.get("data"):
                uparts.append({"inlineData": {"mimeType": img.get("mime", "image/png"), "data": img["data"]}})
            body = {
                "systemInstruction": {"parts": [{"text": load_system_prompt()}]},
                "contents": [{"role": "user", "parts": uparts}],
            }
            keys = GEMINI_KEYS or ([GEMINI_KEY] if GEMINI_KEY else [])
            data_bytes = json.dumps(body).encode("utf-8")
            got = False
            last_err = None
            ki = 0
            retry = 0
            while ki < len(keys) and not got:
                req = urllib.request.Request(
                    url, data=data_bytes,
                    headers={"Content-Type": "application/json", "x-goog-api-key": keys[ki]},
                )
                try:
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        for raw in resp:
                            raw = raw.strip()
                            if not raw or not raw.startswith(b"data:"):
                                continue
                            try:
                                obj = json.loads(raw[5:].strip())
                                for p in obj["candidates"][0]["content"]["parts"]:
                                    if p.get("text"):
                                        got = True
                                        sse(p["text"])
                            except Exception:
                                continue
                    break  # muvaffaqiyat
                except urllib.error.HTTPError as e:
                    body_txt = e.read().decode("utf-8", "ignore")
                    try:
                        last_err = json.loads(body_txt)["error"]["message"][:180]
                    except Exception:
                        last_err = body_txt[:180]
                    if got:
                        break
                    if e.code == 429:               # kvota -> keyingi kalit
                        ki += 1; retry = 0; continue
                    if e.code in (503, 500) and retry < 2:   # band -> shu kalitni qayta
                        retry += 1; time.sleep(1.5 * retry); continue
                    ki += 1; retry = 0; continue    # boshqa xato -> keyingi kalit
                except Exception as e:
                    last_err = str(e)
                    if got:
                        break
                    if retry < 2:
                        retry += 1; time.sleep(1.5); continue
                    ki += 1; retry = 0; continue
            if not got and last_err:
                sse(f"[Gemini band, keyinroq urining: {last_err[:80]}]")
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def stream_research(self):
        """Deep Research: SerpApi qidiruv + Gemini sintezi (manbali javob)."""
        data = self._read_body()
        query = (data.get("prompt") or "").strip()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        def sse(txt):
            payload = json.dumps({"c": txt}, ensure_ascii=False)
            self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
            self.wfile.flush()

        ctx = research_context(query)
        if not ctx:
            sse("Qidiruv natijasi topilmadi (yoki SerpApi kaliti yo'q).")
        elif not GEMINI_KEY:
            sse("Gemini kaliti yo'q.")
        else:
            src_txt = "\n".join(f"[{i+1}] {c['title']}: {c['snippet']} ({c['link']})" for i, c in enumerate(ctx))
            rprompt = (f"Savol: {query}\n\nInternetdan topilgan manbalar:\n{src_txt}\n\n"
                       "Shu manbalar asosida to'liq, aniq va tuzilgan javob ber. "
                       "Muhim faktlarda manba raqamini [1], [2] ko'rinishida ko'rsat. O'zbek tilida yoz.")
            url = ("https://generativelanguage.googleapis.com/v1beta/models/"
                   "gemini-flash-latest:streamGenerateContent?alt=sse")
            body = {
                "systemInstruction": {"parts": [{"text": load_system_prompt()}]},
                "contents": [{"role": "user", "parts": [{"text": rprompt}]}],
            }
            req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json", "x-goog-api-key": GEMINI_KEY})
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=90) as resp:
                        for raw in resp:
                            raw = raw.strip()
                            if not raw or not raw.startswith(b"data:"):
                                continue
                            try:
                                obj = json.loads(raw[5:].strip())
                                for p in obj["candidates"][0]["content"]["parts"]:
                                    if p.get("text"):
                                        sse(p["text"])
                            except Exception:
                                continue
                    break
                except urllib.error.HTTPError as e:
                    if e.code in (503, 429, 500) and attempt < 2:
                        time.sleep(1.5 * (attempt + 1)); continue
                    sse(f"[Xato {e.code}]"); break
                except Exception as e:
                    sse(f"[Ulanish xatosi: {e}]"); break
            # manbalar
            sse("\n\n— Manbalar —\n")
            for i, c in enumerate(ctx):
                if c["link"]:
                    sse(f"[{i+1}] {c['link']}\n")
        try:
            self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def stream_hf(self):
        """Hugging Face router (Llama, DeepSeek...) orqali streaming."""
        data = self._read_body()
        prompt = (data.get("prompt") or "").strip()
        model = data.get("model") or "meta-llama/Llama-3.3-70B-Instruct"
        if not re.match(r"^[A-Za-z0-9._/:-]+$", model):
            model = "meta-llama/Llama-3.3-70B-Instruct"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        def sse(txt):
            self.wfile.write(f"data: {json.dumps({'c': txt}, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()

        if not HF_KEY:
            sse("HF kaliti yo'q.")
        else:
            body = {"model": model, "stream": True, "max_tokens": 1024,
                    "messages": [{"role": "system", "content": load_system_prompt()},
                                 {"role": "user", "content": prompt}]}
            req = urllib.request.Request("https://router.huggingface.co/v1/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={"Authorization": "Bearer " + HF_KEY, "Content-Type": "application/json",
                         "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) MyAI/1.0"})
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    for raw in resp:
                        raw = raw.strip()
                        if not raw or not raw.startswith(b"data:"):
                            continue
                        chunk = raw[5:].strip()
                        if chunk == b"[DONE]":
                            break
                        try:
                            delta = json.loads(chunk)["choices"][0].get("delta", {})
                            if delta.get("content"):
                                sse(delta["content"])
                        except Exception:
                            continue
            except urllib.error.HTTPError as e:
                msg = e.read().decode("utf-8", "ignore")[:150]
                sse(f"[HF xatosi {e.code}: {msg}]")
            except Exception as e:
                sse(f"[HF ulanish xatosi: {e}]")
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def stream_openrouter(self):
        """OpenRouter (GPT-4o va boshqalar) OpenAI-mos streaming."""
        data = self._read_body()
        prompt = (data.get("prompt") or "").strip()
        model = data.get("model") or "openai/gpt-4o-mini"
        if not re.match(r"^[A-Za-z0-9._/:-]+$", model):
            model = "openai/gpt-4o-mini"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        def sse(txt):
            self.wfile.write(f"data: {json.dumps({'c': txt}, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()

        if not OPENROUTER_KEY:
            sse("OpenRouter kaliti yo'q.")
        else:
            body = {"model": model, "stream": True, "max_tokens": 1024,
                    "messages": [{"role": "system", "content": load_system_prompt()},
                                 {"role": "user", "content": prompt}]}
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={"Authorization": "Bearer " + OPENROUTER_KEY, "Content-Type": "application/json",
                         "HTTP-Referer": "https://myai.app", "X-Title": "MyAI"})
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    for raw in resp:
                        raw = raw.strip()
                        if not raw or not raw.startswith(b"data:"):
                            continue
                        chunk = raw[5:].strip()
                        if chunk == b"[DONE]":
                            break
                        try:
                            delta = json.loads(chunk)["choices"][0].get("delta", {})
                            if delta.get("content"):
                                sse(delta["content"])
                        except Exception:
                            continue
            except urllib.error.HTTPError as e:
                msg = e.read().decode("utf-8", "ignore")[:150]
                sse(f"[OpenRouter xatosi {e.code}: {msg}]")
            except Exception as e:
                sse(f"[OpenRouter ulanish xatosi: {e}]")
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
