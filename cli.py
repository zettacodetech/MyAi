#!/usr/bin/env python3
"""
MyAI CLI - terminalda ishlaydigan agent (Claude Code uslubida).
Fayl yaratadi/o'qiydi, buyruq bajaradi. OpenRouter (GPT-4o) tool-calling bilan.
Ishga tushirish: myai   (yoki: myai "vazifa")
"""
import json, os, sys, subprocess, urllib.request, urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent
def load_env():
    env = {}
    f = ROOT / ".env"
    if f.exists():
        for ln in f.read_text().splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#") and "=" in ln:
                k, v = ln.split("=", 1); env[k.strip()] = v.strip()
    return env
ENV = load_env()
GROQ_KEY = os.environ.get("GROQ_KEY") or ENV.get("GROQ_KEY", "")
OR_KEY = os.environ.get("OPENROUTER_KEY") or ENV.get("OPENROUTER_KEY", "")
MODEL = os.environ.get("MYAI_CLI_MODEL", "llama-3.3-70b-versatile")

class C:
    R="\033[0m"; B="\033[1m"; DIM="\033[2m"; GRN="\033[32m"; YEL="\033[33m"
    BLU="\033[34m"; CYN="\033[36m"; MAG="\033[35m"; RED="\033[31m"; GRAY="\033[90m"
def c(t, col): return f"{col}{t}{C.R}" if sys.stdout.isatty() else t

SYSTEM = (
    "Sen MyAI'san — Inomjonning terminaldagi dasturlash agenti (Claude Code kabi). "
    "GAPIRMA — HARAKAT QIL: write_file bilan fayl yarat, run_command bilan buyruq bajar, "
    "read_file/list_dir bilan tekshir. Ish tugagach 1-2 qatorli qisqa xulosa ber. "
    "Kod toza va ishlaydigan bo'lsin. O'zbek tilida javob ber."
)

TOOLS = [
    {"type":"function","function":{"name":"write_file","description":"Faylni yaratadi yoki qayta yozadi. Papkalar avtomatik.",
        "parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}},
    {"type":"function","function":{"name":"read_file","description":"Fayl mazmunini o'qiydi.",
        "parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}},
    {"type":"function","function":{"name":"run_command","description":"Bash buyrug'ini bajaradi va natijasini qaytaradi.",
        "parameters":{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}}},
    {"type":"function","function":{"name":"list_dir","description":"Papkadagi fayllar ro'yxati.",
        "parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":[]}}},
]

def exec_tool(name, args):
    try:
        if name == "write_file":
            p = Path(args["path"]).expanduser(); p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(args.get("content","")); return f"OK: {p} yozildi ({len(args.get('content',''))} belgi)"
        if name == "read_file":
            p = Path(args["path"]).expanduser()
            return p.read_text(errors="ignore")[:15000] if p.is_file() else f"Xato: {p} yo'q"
        if name == "run_command":
            r = subprocess.run(args["command"], shell=True, capture_output=True, text=True, timeout=120)
            return f"[exit {r.returncode}]\n{((r.stdout or '')+(r.stderr or ''))[:8000]}"
        if name == "list_dir":
            p = Path(args.get("path",".")).expanduser()
            return "\n".join(sorted(x.name+("/" if x.is_dir() else "") for x in p.iterdir())) if p.is_dir() else f"Xato: {p} yo'q"
    except subprocess.TimeoutExpired: return "[timeout 120s]"
    except Exception as e: return f"Xato: {e}"
    return "Noma'lum vosita"

def call(messages):
    body = {"model": MODEL, "messages": messages, "tools": TOOLS, "tool_choice": "auto", "max_tokens": 2048}
    if GROQ_KEY:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json", "User-Agent": "Mozilla/5.0 MyAI-CLI/1.0"}
    else:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OR_KEY}", "Content-Type": "application/json",
                   "HTTP-Referer": "https://myai.app", "X-Title": "MyAI CLI", "User-Agent": "Mozilla/5.0 MyAI-CLI/1.0"}
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())["choices"][0]["message"]

def agent(messages):
    for _ in range(30):
        try:
            msg = call(messages)
        except urllib.error.HTTPError as e:
            print(c(f"[Xato {e.code}: {e.read().decode('utf-8','ignore')[:200]}]", C.RED)); return
        except Exception as e:
            print(c(f"[Ulanish xatosi: {e}]", C.RED)); return
        messages.append({"role":"assistant","content":msg.get("content") or "",
                         **({"tool_calls":msg["tool_calls"]} if msg.get("tool_calls") else {})})
        if msg.get("content"): print(c("MyAI: ", C.GRN) + msg["content"])
        tcs = msg.get("tool_calls")
        if not tcs: return
        for tc in tcs:
            fn = tc["function"]["name"]
            try: fa = json.loads(tc["function"].get("arguments") or "{}")
            except: fa = {}
            disp = ", ".join(f"{k}={str(v)[:40]}" for k,v in fa.items())
            print(c(f"  → {fn}({disp})", C.CYN))
            res = exec_tool(fn, fa)
            print(c(f"    {res.splitlines()[0][:100] if res else ''}", C.GRAY))
            messages.append({"role":"tool","tool_call_id":tc["id"],"content":res})
    print(c("[30 qadam chegarasi]", C.YEL))

BANNER = r"""
  __  __       _    ___   CLI
 |  \/  |_   _| |  / \ |  terminaldagi AI agent
 | |\/| | | | | | / _ \|  (Claude Code uslubida)
 |_|  |_|\_, |_|/_/ \_\
         |__/            """

def main():
    if not OR_KEY:
        print(c("OPENROUTER_KEY topilmadi (.env da).", C.RED)); sys.exit(1)
    msgs = [{"role":"system","content":SYSTEM}]
    if len(sys.argv) > 1:
        msgs.append({"role":"user","content":" ".join(sys.argv[1:])}); agent(msgs); return
    print(c(BANNER, C.CYN)); print(c(f"  Model: {MODEL}  ('exit' — chiqish)\n", C.GRAY))
    while True:
        try: u = input(c("Siz: ", C.B+C.BLU)).strip()
        except (EOFError, KeyboardInterrupt): print(c("\nXayr!", C.CYN)); break
        if not u: continue
        if u in ("exit","quit","/exit"): print(c("Xayr!", C.CYN)); break
        msgs.append({"role":"user","content":u}); agent(msgs)

if __name__ == "__main__":
    main()
