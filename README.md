# MyAI — 0 dan qurilgan sun'iy intellekt

Hech qanday tayyor AI, API yoki ML framework (PyTorch/TensorFlow) ISHLATILMAGAN.
Neyron tarmoq — forward pass, backpropagation, o'qitish — hammasi 0 dan, qo'lda yozilgan.
Faqat `numpy` (matritsa hisoblari uchun).

## Tuzilishi
- `model.py`  — neyron til modeli (character-level MLP), 0 dan
- `train.py`  — modelni corpus.txt da o'qitadi -> model.pkl
- `corpus.txt`— o'qitish matni (o'zbekcha)
- `server.py` — backend (pure Python http.server)
- `public/`   — sayt (Claude uslubidagi chat dizayni)
- `model.pkl` — o'qitilgan model og'irliklari

## Ishga tushirish
```bash
./run.sh              # http://localhost:3070
```
Brauzerda **http://localhost:3070** ni oching.

## Qayta o'qitish (aqlliroq qilish)
1. `corpus.txt` ga ko'proq o'zbekcha matn qo'shing (qancha ko'p — shuncha yaxshi).
2. `./venv/bin/python train.py corpus.txt 12000` — qayta o'qitadi.
3. Serverni qayta ishga tushiring.

## Cheklov
Bu KICHIK model (kam matn, kam parametr) — javoblari sodda. Bu ataylab:
maqsad — AI qanday ishlashini 0 dan ko'rsatish. Korpus va model o'lchamini
oshirsangiz, sifat oshadi.
